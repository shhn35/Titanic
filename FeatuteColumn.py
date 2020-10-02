import tensorflow as tf

class CreateFeatureColumn:
  def __init__(self,df,numerical_column_names,categorical_column_names):
    self.df_feature = df
    self.numerical_column_names = numerical_column_names
    self.categorical_column_names = categorical_column_names

  def fc_first_scenario(self):
    '''
    Create naiv feature columns just based on numerical columns
    '''
    feature_column=[]

    # Numerical Columns
    for num_col in self.numerical_column_names:
      feature_column.append(fc.numeric_column(num_col,dtype=tf.float32))

    for cat_col in self.categorical_column_names:
      vocab = self.df_feature[cat_col].unique()
      fc_cat_col = fc.categorical_column_with_vocabulary_list(key=cat_col,vocabulary_list=vocab)

      # Wrap categorical_column with Indicator_column
      feature_column.append(fc.indicator_column(fc_cat_col))
    
    return feature_column
  def fc_second_scenario(self):
    '''
    Creates Tensorflow Feature_column and returns the feature_columns from the Datafram dataset
    This is the SECOND scenario of FC -->
    1- Age is Bucketized fc
    2- Fare is Bucketized fc
    3- Pclass is Numerical fc
    4- Sex is Categorical fc
    '''
    feature_columns = []
    ## Bucketized-columns

    # AGE
    fc_age = tf.feature_column.numeric_column(self.numerical_column_names.pop(0))
    fc_age_bucketized = tf.feature_column.bucketized_column(fc_age,boundaries=[0,1,5,10,15,25,40,55,65])
    feature_columns.append(fc_age_bucketized)
    
    # FARE
    fc_fare = tf.feature_column.numeric_column(self.numerical_column_names.pop(0))
    fc_fare_bucketized = tf.feature_column.bucketized_column(fc_fare,boundaries=[0,10,25,56,70])
    feature_columns.append(fc_fare_bucketized)

    # Numerica Features Column
    for feature_name in self.numerical_column_names:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.int8))
    
    # Categorical Features Column with Vocabulary
    for feature_name in self.categorical_column_names:
        feature_vocabulary_list = self.df_feature[feature_name].unique()

        fc_cat_feature_column_with_vocab = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,feature_vocabulary_list)

        ## Wrap the categorical column with Indicator_column in case of having alimitted vocabulary list
        feature_columns.append(tf.feature_column.indicator_column(fc_cat_feature_column_with_vocab))

    return feature_columns

  def fc_third_scenario(self):
    '''
    Creates Tensorflow Feature_column and returns the feature_columns from the Datafram dataset
    This is the SECOND scenario of FC -->
    1- Age is Bucketized fc
    2- Fare is Bucketized fc
    3- Pclass is Bucketized fc
    4- Sex is Categorical fc
    5- Cross feature of Fare and Pclass
    6- cross feature of Fare and Sex
    7- Cross feature of Age and Sex
    '''
    feature_columns = []

    ## Bucketized-columns

    # AGE
    fc_age = tf.feature_column.numeric_column(self.numerical_column_names.pop(0))
    fc_age_bucketized = tf.feature_column.bucketized_column(fc_age,boundaries=[0,1,5,10,15,25,40,55,65])
    feature_columns.append(fc_age_bucketized)
    
    # FARE
    fc_fare = tf.feature_column.numeric_column(self.numerical_column_names.pop(0))
    fc_fare_bucketized = tf.feature_column.bucketized_column(fc_fare,boundaries=[0,10,25,56,70])
    feature_columns.append(fc_fare_bucketized)

    #CLASS
    fc_class = tf.feature_column.numeric_column(self.numerical_column_names.pop(0))
    fc_class_bucketized = tf.feature_column.bucketized_column(fc_class,boundaries=[1,2])
    feature_columns.append(fc_class_bucketized)

    ## Numerica Features Column
    for feature_name in self.numerical_column_names:
        feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.int8))
    
    ## Categorical Features Column with Vocabulary
    
    #SEX
    sex_column_name = self.categorical_column_names.pop(0)
    sex_vocab_list = self.df_feature[sex_column_name].unique()
    fc_sex = tf.feature_column.categorical_column_with_vocabulary_list(sex_column_name,sex_vocab_list)
    feature_columns.append(tf.feature_column.indicator_column(fc_sex))

    for feature_name in self.categorical_column_names:
        feature_vocabulary_list = self.dataframe[feature_name].unique()

        fc_cat_feature_column_with_vocab = tf.feature_column.categorical_column_with_vocabulary_list(feature_name,feature_vocabulary_list)

        ## Wrap the categorical column with Indicator_column in case of having alimitted vocabulary list
        feature_columns.append(tf.feature_column.indicator_column(fc_cat_feature_column_with_vocab))

    # # Cross feature of 'age' and 'sex'

    # SEX and AGE
    fc_sex_age_cross_feature = tf.feature_column.crossed_column([fc_age_bucketized,fc_sex],hash_bucket_size=20)
    feature_columns.append(tf.feature_column.indicator_column(fc_sex_age_cross_feature))

    # SEX and FARE
    fc_sex_fare_cross_feature = tf.feature_column.crossed_column([fc_fare_bucketized,fc_sex],hash_bucket_size=12)
    feature_columns.append(tf.feature_column.indicator_column(fc_sex_fare_cross_feature))

    # FARE and CLASS
    fc_fare_class_cross_feature =tf.feature_column.crossed_column([fc_fare_bucketized,fc_class_bucketized],hash_bucket_size=12)
    feature_columns.append(tf.feature_column.indicator_column(fc_fare_class_cross_feature))

    return feature_columns
  
