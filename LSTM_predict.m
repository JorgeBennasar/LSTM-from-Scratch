function [Y_pred, A] = LSTM_predict(X, param)

[A, Y_aux, ~] = LSTM_forward_prop(X, param);
Y_pred = Y_aux;

end