import tensorflow as tf
# from tensorflow.keras import optimizers as optimazers
import math
import csv

class TfModels:
  def __init__(self,df_train,y_train,df_test,y_test,df_predict,batch_size,shuffle,lable_class_vocab,feature_columns,network_size,y_predict=None):
    self.df_train = df_train
    self.y_train = y_train
    self.df_test = df_test
    self.y_test = y_test
    self.df_predict = df_predict
    self.y_predict = y_predict
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.feature_columns = feature_columns
    self.lable_class_vocab = lable_class_vocab
    self.network_size = network_size
  
  def __print_prediction_fault(self,sample_index,wrong_sample_index,actual,prediction):
    print('{}th Wrong result occures at {}th sample ->\t Actual output is: "{}"\t Prediction result is "{}"'.format(
        wrong_sample_index,sample_index,actual,prediction
    ))

  def __calc_prediction_result(self,y_predict,prediction_output,show_pred_result):  
    sample_index = 0
    wrong_sample_index = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    correct_pred = 0
    for i in y_predict:    
      if(i == 0): 
        # negetive actual output
        if(prediction_output[sample_index] == 0):
          # TRUE NEGATIVE
          tn += 1
          correct_pred += 1
        else:
          # FALSE NEGATIVE
          fn += 1
          wrong_sample_index +=1
          if show_pred_result:
            self.__print_prediction_fault(sample_index,wrong_sample_index,i,prediction_output[sample_index])
      else:
        # positive actual output
        if(prediction_output[sample_index] == 1):
          # TRUE POSITIVE
          tp += 1
          correct_pred += 1
        else:
          # FALSE POSITIVE
          fp += 1
          wrong_sample_index +=1
          if show_pred_result:
            self.__print_prediction_fault(sample_index,wrong_sample_index,i,prediction_output[sample_index])
      sample_index += 1

    pred_acc = (correct_pred / len(y_predict))

    return tp,fp,tn,fn,correct_pred,pred_acc

  def __save_prediction_result(self,file_name,prediction_output):
    with open(file='dataset/output/' + file_name,
              mode='w',
              newline='') as pred_out_file:
      wr = csv.writer(pred_out_file,quoting=csv.QUOTE_ALL)
      wr.writerow(prediction_output)
  
  def __get_hidden_units(self,len_fc,network_size):
    '''
    Dynamically generates the number of nodes in each layers of NN
    "network_size" indicates how dense the nodes should be. possible values are: "tiny,small,medium,large"
    '''
    hidden_units = []
    node_power = 0
    fc_coefficient = 0

    if network_size=='large':
      node_power = 4
      fc_coefficient = 100
    elif network_size=='medium':
      node_power = 3
      fc_coefficient = 50
    elif network_size=='small':
      node_power = 3
      fc_coefficient = 25
    elif network_size=='tiny':
      node_power = 2
      fc_coefficient = 10

    while node_power <= int(math.log2(len_fc * fc_coefficient)):
      hidden_units.append(2**node_power)
      node_power += 1

    hidden_units.reverse()
    return hidden_units

  def __input_fn(self,df_feature, lables, batch_size, shuffle=False):
    buffer_size=1024
    
    ds_out = tf.data.Dataset.from_tensor_slices((dict(df_feature),lables))

    if(shuffle):
      ds_out = ds_out.shuffle(buffer_size=buffer_size).repeat()

    return ds_out.batch(batch_size=batch_size)

  def __input_fn_predict(self,df_feature,batch_size):
    return tf.data.Dataset.from_tensor_slices((dict(df_feature))).batch(batch_size=batch_size)
    
  def tf_DNN_Classifier(self,optimizer,show_best_epoch_trend,max_epoch,min_epoch=0,epoch_step=0,show_pred_result=True,save_pred_result=True):      
    eval_accuracy = []
    pred_accuracy = []
    
    if show_best_epoch_trend:
      show_pred_result = False
      save_pred_result = False
    else:
      min_epoch = max_epoch

    if not show_best_epoch_trend:
      eval_acc,pred_acc = self.__run_tf_DNN_Classifier(max_epoch,optimizer,show_pred_result,save_pred_result)
      
      eval_accuracy.append(eval_acc)
      if pred_acc is not None:
        pred_accuracy.append(pred_acc)

    
    return eval_accuracy,pred_accuracy

  def __run_tf_DNN_Classifier(self,max_epoch,optimizer,show_pred_result,save_pred_result):
    '''
    The first model which is based on TF.esstimator.dnnClassifier
    '''
    dnn_classifier = tf.estimator.DNNClassifier(hidden_units=lambda :self.__get_hidden_units(len(self.feature_columns),self.network_size),
                                            feature_columns = self.feature_columns,
                                            n_classes = len(self.lable_class_vocab),
                                            # label_vocabulary = lable_class_vocab
                                            # optimizer = 'Adagrad'
                                            )
    
    #Train
    dnn_classifier.train(input_fn= lambda: self.__input_fn(self.df_train,self.y_train,self.batch_size,self.shuffle),
                    steps = max_epoch
                     )
    
    #Evaluate
    eval_result = dnn_classifier.evaluate(input_fn= lambda: self.__input_fn(self.df_test,self.y_test,self.batch_size))
    if show_pred_result:
      print('Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Prediction
    prediction_result = dnn_classifier.predict(input_fn=lambda: self.__input_fn_predict(self.df_predict,self.batch_size))    

    # determining all prediction results
    prediction_output = []
    prob =[]
    i=1
    for predict_dict in prediction_result:      
      class_id = predict_dict['class_ids'][0]
      probability = predict_dict['probabilities'][class_id]

      # print('{}: The prediction is "{}->{}" with the probability of {:.1f}%'.format(
      #     i,class_id,lable_class_vocab[class_id],100 * probability
      # ))

      prediction_output.append(class_id)
      i+=1


    if save_pred_result:
      self.__save_prediction_result(file_name = 'prediction_result.csv',prediction_output=prediction_output)
      print('Prediction Result has sucessfully been saved!')

    # demonstrate the prediction result in case of having y_predict
    pred_acc = 'Showing Prediction result is not possible!'
    if self.y_predict is not None:
      tp,fp,tn,fn,correct_pred,pred_acc = self.__calc_prediction_result(self.y_predict,prediction_output,show_pred_result)

      if show_pred_result:
        print(prediction_output)
        print('All prediction samples: {}\nAll Correct_predictions: {}\nPrediction Accuracy: {}\nTrue Positive: {}\nTrue Negative: {}\nFalse Posotive: {}\nFalse Negative: {}\n'.format(
            len(self.y_predict),
            correct_pred,
            pred_acc,
            tp,tn,fp,fn
        ))
      
    return eval_result['accuracy'],pred_acc

    