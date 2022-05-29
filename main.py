import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
import time
import logging
from datetime import datetime
import torchvision
from copy import deepcopy
import torchvision.transforms as T
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
torch.set_default_tensor_type(torch.DoubleTensor)

writer = SummaryWriter(log_dir=f"./runs/{datetime.now()}", flush_secs=10)


def make_grid(tensor):
    img_num = tensor.shape[0]
    # return torch.cat((torch.cat([t for t in tensor[:img_num//2]], 0),
    #  torch.cat([t for t in tensor[img_num//2:]], 0)), 1)
    return torch.cat([t for t in tensor[:img_num]], 0)

def form_reconstructed_images(images, masks):
    reconstructed_images = []
    for image,mask in zip(images, masks):
        image_height, image_width = image.shape
        dimension_chunk_num = int(mask.shape[0]**0.5)
        chunk_height, chunk_width = image_height//dimension_chunk_num, image_width//dimension_chunk_num
        M = torch.argmax(mask, dim=1)

        reconstructed_image = torch.zeros_like(image)
        for reconstructed_idx, original_idx in enumerate(M):
            reconstructed_i, reconstructed_j = reconstructed_idx//dimension_chunk_num, reconstructed_idx%dimension_chunk_num
            original_i, original_j = original_idx//dimension_chunk_num, original_idx%dimension_chunk_num

            reconstructed_image[reconstructed_i*chunk_height:(reconstructed_i+1)*chunk_height,
            reconstructed_j*chunk_width:(reconstructed_j+1)*chunk_width
            ] = \
                image[original_i*chunk_height:(original_i+1)*chunk_height,
            original_j*chunk_width:(original_j+1)*chunk_width
            ]
        reconstructed_images.append(reconstructed_image)
    return torch.stack(reconstructed_images)


class Solver():
    def __init__(self, model, linear_cca, linear_outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.linear_outdim_size = linear_outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        Mregs = []
        corrs = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            # print(f"Batch Number: {self.batch_size}")
            for idx, batch_idx in enumerate(batch_idxs):
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2, M = self.model(batch_x1, batch_x2)

                loss, corr, Mreg = self.loss(o1, o2, M)
                train_losses.append(loss.item())
                corrs.append(corr.item())
                Mregs.append(Mreg.item())
                loss.backward()

                # Log to tensorboard
                writer.add_scalars('TrainBatchStats',
                                   {
                                       'batch_total_loss': loss.item(),
                                       'batch_cor': corr.item(),
                                       'batch_mreg_loss': Mreg.item()
                                   },
                                   epoch*len(batch_idxs)+idx+1
                                   )

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            corr = np.mean(corrs)
            Mreg = np.mean(Mregs)

            writer.add_scalars('TrainEpochStats', {
                'epoch_total_loss': train_loss,
                'epoch_cor': corr,
                'epoch_mreg_loss': Mreg
            }, epoch+1)

            with torch.no_grad():
                input_1 = x1[:30, :]
                input_2 = x2[:30, :]
                output_1, output_2, M = self.model(input_1, input_2)
                reconstructed_images = form_reconstructed_images(input_1.view(-1,28,28), M)
                writer.add_image('train_0/first_view_original',
                                 make_grid(input_1.view(-1, 28, 28)), epoch+1, dataformats='HW')
                writer.add_image('train_0/second_view_original',
                                 make_grid(input_2.view(-1, 28, 28)), epoch+1, dataformats='HW')
                writer.add_image('train_0/second_view_reconstructed',
                                 make_grid(reconstructed_images.view(-1, 28, 28)), epoch+1, dataformats='HW')
                writer.add_image('train_0/first_view_transformed_feature',
                                 make_grid(output_1.view(-1, 4, 4)), epoch+1, dataformats='HW')
                writer.add_image('train_0/first_view_transformed_feature_with_M',
                                 make_grid(torch.matmul(M, output_1.view(-1,16,1)).view(-1, 4, 4)), epoch+1, dataformats='HW')                 
                writer.add_image('train_0/second_view_transformed_feature',
                                 make_grid(output_2.view(-1, 4, 4)), epoch+1, dataformats='HW')
                writer.add_image('train_0/permutation_matrix',
                                 make_grid(M.view(-1, 16, 16)), epoch+1, dataformats='HW')

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f} Corr: {:.4f} Mreg: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss, outputs = self.test(vx1, vx2)
                    writer.add_scalar("Val/EpochTotalLoss", val_loss, epoch+1)

                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        # self.logger.info(
                        #     "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    # else:
                        # self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                        #     epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss, corr, Mreg))

        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss, outputs = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss, outputs = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.linear_outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            Mvalues = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2, M = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                Mvalues.append(M)
                loss, corr, M_reg = self.loss(o1, o2, M)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy(),
                   torch.cat(Mvalues, dim=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':
    ############
    # Parameters Section

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the final learned features
    save_to = './new_features.gz'

    # size of the input for view 1 and view 2
    input_shape1 = (28, 28)
    input_shape2 = (28, 28)

    # output size for the linear cca
    linear_outdim_size = 10

    # the number of eigen values to check in the correlation
    k_eigen_check_num = 16

    # the parameters for training the network
    learning_rate = 1e-4
    epoch_num = 500
    batch_size = 256

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False
    # end of parameters section
    ############

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    data1 = load_data('./noisymnist_view1.gz', convert_to_image=True)
    # data2 = load_data('./noisymnist_view2.gz', convert_to_image=True)
    data2 = deepcopy(data1)

    transform = T.Compose([
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation((-180,180)),
        T.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.8, 1))
        ])
    for sub_data_index, sub_data in enumerate(data2):
        data2[sub_data_index] = (transform(sub_data[0]), sub_data[1])

    # Building, training, and producing the new features by DCCA
    model = DeepCCA(input_shape1,
                    input_shape2,
                    k_eigen_check_num,
                    use_all_singular_values,
                    device=device).double()

    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model,
                    l_cca,
                    linear_outdim_size,
                    epoch_num,
                    batch_size,
                    learning_rate,
                    reg_par,
                    device=device)
    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]

    solver.fit(train1, train2, val1, val2, test1, test2)

    set_size = [0,
                train1.size(0),
                train1.size(0) + val1.size(0),
                train1.size(0) + val1.size(0) + test1.size(0)]
    loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat(
        [train2, val2, test2], dim=0), apply_linear_cca)
    new_data = []

    for idx in range(3):
        new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :],
                         outputs[1][set_size[idx]:set_size[idx + 1], :],
                         data1[idx][1]])

    # Training and testing of SVM with linear kernel on the view 1 with new features
    [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 1 (test data) is:", test_acc*100.0)

    # Saving new features in a gzip pickled file specified by save_to
    print('saving new features ...')
    f1 = gzip.open(save_to, 'wb')
    thepickle.dump(new_data, f1)
    f1.close()
    d = torch.load('checkpoint.model')
    solver.model.load_state_dict(d)
    solver.model.parameters()
