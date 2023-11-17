import unittest
from unittest import mock
from neural_network import NeuralNetwork
from neural_network import predict_message_mood


class MyNeuralNetwork(unittest.TestCase):
    def setUp(self)->None:
        self.model = NeuralNetwork()

    def test_predict_message_mood(self)->None:
        with mock.patch('neural_network.NeuralNetwork.predict') as mock_predict:
            mock_predict.side_effect = [0.05, 0.5, 0.8]
            self.assertEqual(predict_message_mood('Чапаев и пустота', self.model, bad=0.1, good=0.21), 'неуд')
            self.assertEqual(predict_message_mood('Тень на Иннсмутом', self.model, bad=0.4, good=0.6), 'норм')
            self.assertEqual(predict_message_mood('Геральд', self.model, bad=0.6, good=0.75), 'отл')


if __name__ == '__main__':
    unittest.main()