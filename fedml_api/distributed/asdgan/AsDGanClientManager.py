import logging, sys, os


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message

from .message_define import MyMessage
from .utils import transform_list_to_tensor


class AsDGanClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.epochs
        self.epoch_idx = 0
        self.iter_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_FAKE_DATA_TO_CLIENT,
                                              self.handle_message_receive_fake_data_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_STOP,
                                              self.handle_message_stop)

    def handle_message_init(self, msg_params):
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info('[Client {0}] Initial label samples to central server'.format(client_index))

        self.trainer.update_dataset(int(client_index))

        self.epoch_idx = 0
        local_sample_number, keys, labels = self.trainer.upload_labels()
        # logging.info('[Client {0}] labels: size {1} and type {2}'.format(client_index, len(labels), labels[0].dtype))
        self.send_label_to_server(0, local_sample_number, keys, labels)

    def handle_message_receive_fake_data_from_server(self, msg_params):
        key_samples = msg_params.get(MyMessage.MSG_ARG_KEY_KEY_SAMPLES)
        fake_samples = msg_params.get(MyMessage.MSG_ARG_KEY_FAKE_SAMPLES)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        n_rest_iter = msg_params.get(MyMessage.MSG_ARG_KEY_EPOCH_INDEX)
        transform_params = msg_params.get(MyMessage.MSG_ARG_KEY_FAKE_SAMPLES_AUG_PARAMS)

        # logging.info(f'$$$[Client {client_index}] receive data client key_samples:{key_samples}, '
        #              f'fake_samples:{len(fake_samples)} tsfm:{len(transform_params)}')

        # logging.info('[Client {0}] Received fake samples from central server'.format(client_index))

        self.__train(client_index, key_samples, fake_samples, n_rest_iter, transform_params)

    def handle_message_stop(self, msg_params):
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info('[Client {0}] Received stop signal from server'.format(client_index))
        self.finish()

    def send_label_to_server(self, receive_id, local_sample_num, keys, labels):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_LABEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_KEY_SAMPLES, keys)
        message.add_params(MyMessage.MSG_ARG_KEY_LABEL_SAMPLES, labels)
        self.send_message(message)

    def send_grad_to_server(self, receive_id, grad_fake, local_sample_num, weights, train_evaluation_metrics, test_evaluation_metrics):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_SAMPLES, grad_fake)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_EVALUATION_METRICS, train_evaluation_metrics)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_EVALUATION_METRICS, test_evaluation_metrics)
        self.send_message(message)

    def __train(self, client_index, key_samples, fake_samples, n_rest_iter, transform_params):

        train_evaluation_metrics = test_evaluation_metrics = None
        weights = None
        if (self.iter_idx+1) % 100 == 0:
            # logging.info("####### Testing Global Params ########### round_id = {}".format(self.epoch_idx))
            # train_evaluation_metrics, test_evaluation_metrics = self.trainer.test(self.epoch_idx)
            logging.info("[Client {0}] ####### Training ########### epoch_idx={1} iter_idx={2}".format(client_index, self.epoch_idx, self.iter_idx))

        grad, local_sample_num, train_evaluation_metrics = self.trainer.train(key_samples, fake_samples, transform_params)

        # logging.info("[Client {0}] ####### Testing Client Params ########### round_id = {1}".format(client_index, self.epoch_idx))
        # train_evaluation_metrics, test_evaluation_metrics = self.trainer.test(self.epoch_idx)
        self.iter_idx += 1
        if n_rest_iter == 1:
            self.epoch_idx += 1
            self.iter_idx = 0
            self.trainer.update_lr()  # update learning rates at the end of every epoch.
            weights = self.trainer.get_model()

        self.send_grad_to_server(0, grad, local_sample_num, weights, train_evaluation_metrics, test_evaluation_metrics)
