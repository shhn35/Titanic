# from df_dataset_overview import dataframe_overview as df_overview
import dataset as ds
import FeatuteColumn as fc
import TfModels 


def main():
    # Constants
    lable_column_name = 'Survived'
    lable_class_vocab = ['SANK','SURVIVED']
    numerical_column_names = ['Age','Fare','Pclass']
    categorical_column_names = ['Sex']

    data_set_type = 'kaggle' # should be either 'kaggle' or 'google'
    max_epoch = 70000
    batch_size = 256
    shuffle = True
    test_size = 80
    network_size='medium'
    fc_scenario_num = 2

    relevant_columns = numerical_column_names + categorical_column_names
    relevant_columns.append(lable_column_name)

    # Models for prediction on Titanic survival
    df_train,y_train,df_test,y_test,df_predict,y_predict = ds.read_dataset(test_size=test_size,data_set_type=data_set_type,relevant_columns=relevant_columns,lable_column_name=lable_column_name)


    
    create_feature_column = fc.CreateFeatureColumn(df=df_train.append(df_test.append(df_predict)),
                                            numerical_column_names=numerical_column_names,
                                            categorical_column_names=categorical_column_names)

    feature_column = None
    if fc_scenario_num == 2:
        feature_column = create_feature_column.fc_second_scenario()
    elif fc_scenario_num == 3:
        feature_column = create_feature_column.fc_third_scenario()
    else:
        feature_column = create_feature_column.fc_first_scenario()

    for f in feature_column:
        print(f)

    print('{} Featutes have been created based on Scenario {}'.format(
        len(feature_column),
        fc_scenario_num
        ))


    tf_model = TfModels.TfModels(df_train,y_train,
                        df_test,y_test,
                        df_predict,
                        batch_size,
                        shuffle,
                        lable_class_vocab,
                        feature_column,
                        network_size,
                        y_predict)

    eval_accuracy,pred_accuracy = tf_model.tf_DNN_Classifier(optimizer = None,
                                                            show_best_epoch_trend = False,
                                                            max_epoch = max_epoch,
                                                            show_pred_result = True,
                                                            save_pred_result = True)


    print('\nEvaluation Accuracy is {}\nPrediction Accuracy is: {}'.format(
        eval_accuracy,pred_accuracy
    ))



if __name__ == "__main__":
    main()