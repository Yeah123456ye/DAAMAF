# import APIs
import utils
import engine
# import ETest
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os  # 用于文件操作

config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class experiment():
    def __init__(self, fold_idx, task):
        self.fold_idx = fold_idx
        self.task = task

        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.best_weights = None

        # Learning schedules
        self.num_epochs = 200 # 200
        self.num_batches = 5
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=1000,
                                                                 decay_rate=.96, staircase=False)

        # Loss control hyperparameter
        self.alpha_rec_SNP = 1.0
        self.alpha_gen = 1.0
        self.alpha_dis = 1.0
        self.alpha_se = 1.0
        self.alpha_clf = 1.0

    def training(self):
        print(f'Start Training, Fold {self.fold_idx}')

        # Load dataset
        X_MRI_train, X_PET_train, E_SNP_train, C_demo_train, Y_train, S_train, \
        X_MRI_val, X_PET_val, E_SNP_val, C_demo_val, Y_val, S_val, \
        X_MRI_test, X_PET_test, E_SNP_test, C_demo_test, Y_test, S_test = utils.load_dataset(self.fold_idx, self.task)
        N_o = Y_train.shape[-1]
        # Call optimizers
        opt_rec = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_gen = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_dis = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_se = tf.keras.optimizers.Adam(learning_rate=self.lr)
        opt_clf = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Call ENGINE framework
        model = engine.engine(N_o=N_o)

        num_iters = int(Y_train.shape[0]/self.num_batches)

        for epoch in range(self.num_epochs):
            L_rec_per_epoch = 0
            L_gen_per_epoch = 0
            L_dis_per_epoch = 0
            L_se_per_epoch = 0
            L_clf_per_epoch = 0

            # Randomize the training dataset
            rand_idx = np.random.permutation(Y_train.shape[0])
            X_MRI_train = X_MRI_train[rand_idx, ...]
            X_PET_train = X_PET_train[rand_idx, ...]
            E_SNP_train = E_SNP_train[rand_idx, ...]
            C_demo_train = C_demo_train[rand_idx, ...]
            Y_train = Y_train[rand_idx, ...]

            for batch in range(num_iters):
                # Sample a minibatch
                xb_MRI = X_MRI_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                xb_PET = X_PET_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                eb_SNP = E_SNP_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)
                cb_demo = C_demo_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...]
                yb_clf = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, ...].astype(np.float32)

                # Estimate gradient and loss
                with tf.GradientTape() as tape_rec, tf.GradientTape() as tape_gen, tf.GradientTape() as tape_dis, \
                     tf.GradientTape() as tape_se, tf.GradientTape() as tape_clf:

                    # SNP representation module
                    mu, log_sigma_square = model.encode(x_SNP=eb_SNP)
                    zb_SNP = model.reparameterize(mean=mu, logvar=log_sigma_square)
                    eb_SNP_hat_logit = model.decode(z_SNP=zb_SNP)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=eb_SNP_hat_logit, labels=eb_SNP)
                    log_prob_eb_SNP_given_zb_SNP = -tf.math.reduce_sum(cross_ent, axis=1)
                    log_prob_zb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=0., logvar=0.)
                    log_q_zb_given_eb_SNP = utils.log_normal_pdf(sample=zb_SNP, mean=mu, logvar=log_sigma_square)

                    # Reconstruction loss
                    L_rec = -tf.math.reduce_mean(log_prob_eb_SNP_given_zb_SNP + log_prob_zb_SNP - log_q_zb_given_eb_SNP)
                    L_rec *= self.alpha_rec_SNP

                    # multi-modal imaging fusion
                    zb_MRI,c1,rec_MRI = model.encode_MRI(x_MRI=xb_MRI)
                    zb_PET,c2,rec_PET = model.encode_PET(x_PET=xb_PET)
                    MRI_PET, PET_MRI = model.cross_modal_attention(rec_MRI, rec_PET)
                    MRI = tf.add(MRI_PET, zb_MRI)
                    PET = tf.add(PET_MRI, zb_PET)
                    joint_feature = tf.multiply(MRI, PET)

                    # generative network
                    xb_MRI_fake, ab = model.generate(z_SNP=zb_SNP,c_demo=cb_demo)
                    real_output = model.discriminate(x_MRI_real_or_fake=joint_feature)
                    fake_output = model.discriminate(x_MRI_real_or_fake=xb_MRI_fake)

                    # Least-Square GAN loss
                    L_gen = tf.keras.losses.MSE(tf.ones_like(fake_output), fake_output)
                    L_gen *= self.alpha_gen

                    L_dis = tf.keras.losses.MSE(tf.ones_like(real_output), real_output) \
                            + tf.keras.losses.MSE(tf.zeros_like(fake_output), fake_output)
                    L_dis *= self.alpha_dis

                    mask1 = create_diag_mask(c1)
                    c1_no_diag = c1 * mask1

                    mask2 = create_diag_mask(c2)
                    c2_no_diag = c2 * mask2

                    loss_se_mri = tf.norm(tf.matmul(c1_no_diag, zb_MRI) - zb_MRI, ord='fro', axis=[-2, -1]) ** 2
                    loss_se_pet = tf.norm(tf.matmul(c2_no_diag, zb_PET) - zb_PET,ord='fro', axis=[-2, -1]) ** 2
                    loss_se_reg = tf.norm(c1_no_diag, ord='fro', axis=[-2, -1]) ** 2 + tf.norm(c2_no_diag, ord='fro', axis=[-2, -1]) ** 2
                    L_se = (loss_se_mri + loss_se_pet) + loss_se_reg
                    L_se *= self.alpha_se

                    # Diagnostician module
                    yb_clf_hat, sb_reg_hat = model.diagnose(x_MRI=joint_feature, a=ab, apply_logistic_activation=True)

                    # Classification loss
                    L_clf = tfa.losses.sigmoid_focal_crossentropy(yb_clf, yb_clf_hat)
                    L_clf *= self.alpha_clf

                # Apply gradients
                var = model.trainable_variables
                theta_Q = [var[0], var[1], var[2], var[3]]
                theta_P = [var[4], var[5], var[6], var[7]]
                theta_MRI_Encoder = [var[8], var[9], var[10], var[11]]
                theta_PET_Encoder = [var[12], var[13], var[14], var[15]]
                theta_G = [var[16], var[17], var[18], var[19]]
                theta_D = [var[20], var[21]]
                theta_C_share = [var[22], var[23]]
                theta_C_clf = [var[24], var[25]]

                grad_rec = tape_rec.gradient(L_rec, theta_Q + theta_P)
                opt_rec.apply_gradients(zip(grad_rec, theta_Q + theta_P))
                L_rec_per_epoch += np.mean(L_rec)

                grad_gen = tape_gen.gradient(L_gen, theta_Q + theta_G)
                opt_gen.apply_gradients(zip(grad_gen, theta_Q + theta_G))
                L_gen_per_epoch += np.mean(L_gen)

                grad_dis = tape_dis.gradient(L_dis, theta_D)
                opt_dis.apply_gradients(zip(grad_dis, theta_D))
                L_dis_per_epoch += np.mean(L_dis)

                grad_se = tape_se.gradient(L_se, theta_MRI_Encoder + theta_PET_Encoder)
                opt_se.apply_gradients(zip(grad_se, theta_MRI_Encoder + theta_PET_Encoder))
                L_se_per_epoch += np.mean(L_se)

                grad_clf = tape_clf.gradient(L_clf, theta_G + theta_MRI_Encoder + theta_PET_Encoder + theta_C_share + theta_C_clf)
                opt_clf.apply_gradients(zip(grad_clf, theta_G + theta_MRI_Encoder + theta_PET_Encoder + theta_C_share + theta_C_clf))
                L_clf_per_epoch += np.mean(L_clf)

            L_rec_per_epoch /= num_iters
            L_gen_per_epoch /= num_iters
            L_dis_per_epoch /= num_iters
            L_se_per_epoch /= num_iters
            L_clf_per_epoch /= num_iters

            mu_val, log_sigma_square_val = model.encode(E_SNP_val)
            Z_SNP_val = model.reparameterize(mu_val, log_sigma_square_val)
            zb_MRI_val, _, rec_mri_val = model.encode_MRI(x_MRI=X_MRI_val)
            zb_PET_val, _, rec_pet_val = model.encode_PET(x_PET=X_PET_val)
            MRI_PET_val, PET_MRI_val = model.cross_modal_attention(rec_mri_val, rec_pet_val)
            MRI_val = tf.add(MRI_PET_val, zb_MRI_val)
            PET_val = tf.add(PET_MRI_val, zb_PET_val)
            joint_feature_val = tf.multiply(MRI_val, PET_val)
            _, A_val = model.generate(Z_SNP_val, C_demo_val)
            y_val_hat, _ = model.diagnose(joint_feature_val, A_val, True)

            # best val performance
            val_auc = roc_auc_score(Y_val, y_val_hat)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.best_weights = model.get_weights()

        # Results
        model.set_weights(self.best_weights)
        mu, log_sigma_square = model.encode(E_SNP_test)
        Z_SNP_test = model.reparameterize(mu, log_sigma_square)
        zb_MRI,c1,rec_mri = model.encode_MRI(x_MRI=X_MRI_test)
        zb_PET,c2,rec_pet = model.encode_PET(x_PET=X_PET_test)
        MRI_PET, PET_MRI = model.cross_modal_attention(rec_mri, rec_pet)
        MRI = tf.add(MRI_PET, zb_MRI)
        PET = tf.add(PET_MRI, zb_PET)
        joint_feature = tf.multiply(MRI, PET)
        _, A_test = model.generate(Z_SNP_test, C_demo_test)
        Y_test_hat, _ = model.diagnose(joint_feature, A_test, True)
        return Y_test, Y_test_hat


def create_diag_mask(matrix):
    batch_size = tf.shape(matrix)[0]
    num_samples = matrix.shape[-1]
    eye = tf.eye(num_samples, batch_shape=[batch_size])  # [batch, N, N]
    return 1.0 - eye

def calculate_accuracy(y_pred, y_true):
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    correct_predictions = np.sum(y_pred_labels == y_true_labels)
    accuracy = correct_predictions / len(y_true_labels)
    return accuracy

task = ['CN', 'AD']
auc_list = []
acc_list = []
for fold in range(5):  # 5-fold CV
    exp = experiment(fold + 1, ['CN', 'AD'])
    Y_test, Y_test_hat = exp.training()
    AUC = roc_auc_score(Y_test, Y_test_hat)
    print(f'Test AUC: {AUC:>.4f}')
    acc = calculate_accuracy(Y_test_hat, Y_test)
    print(f'Test ACC: {acc:>.4f}')
    auc_list.append(AUC)
    acc_list.append(acc)
mean_auc = np.mean(auc_list)
std_auc = np.std(auc_list)
mean_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
print(f'Mean AUC: {mean_auc:>.4f} '
      f'Mean ACC: {mean_acc:>.4f} ')
print(f'Std AUC: {std_auc:>.4f} '
      f'Std ACC: {std_acc:>.4f} ')