import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.cluster import KMeans
from src.utils import save_object


from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")

        
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                                'Type_of_vehicle', 'Festival', 'City']
            numerical_cols = ['Delivery_person_ID', 'Delivery_person_Age', 'Delivery_person_Ratings',
                              'Vehicle_condition', 'multiple_deliveries', 'Date', 'Month',
                              'Ordered_Hour', 'Ordered_Min', 'Order_pick_Hour', 'Order_pick_Min',
                              ]
            
            # Define the custom ranking for each ordinal variable
            weather_map={"Sandstorms":6,"Stormy":5,"Fog":4,"Windy":3,"Cloudy":2,"Sunny":1}
            traffic_map={"Jam":4,"High":3,"Medium":2,"Low":1}
            order_map={"Meal":4,"Buffet":3,"Snack":2,"Drinks":1}
            vehicle_map={"motorcycle":1,"scooter":2,"electric_scooter":3,"bicycle":4}
            festival_map={"No":1,"Yes":2}
            city_map={"Metropolitian":3,"Urban":2,"Semi-Urban":1}

            logging.info("Pipeline Initiated")

            numeric_transformer = StandardScaler()
            ord_transformer = OrdinalEncoder()
            preprocessor = ColumnTransformer(
                [
                ("Ordinalencoder", ord_transformer, categorical_cols),
                ("StandardScaler", numeric_transformer, numerical_cols)       
                ]
            )

            return preprocessor
        
            logging.info("Pipeline Completed")



        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            categorical_columns = train_df.select_dtypes(include="object").columns
            numerical_columns = train_df.select_dtypes(exclude="object").columns

            categorical_columns = test_df.select_dtypes(include="object").columns
            numerical_columns = test_df.select_dtypes(exclude="object").columns

            cat_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
            cat_imputer = cat_imputer.fit(train_df[categorical_columns])
            train_df[categorical_columns] = cat_imputer.transform(train_df[categorical_columns])

            num_imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            num_imputer = num_imputer.fit(train_df[numerical_columns])
            train_df[numerical_columns] = num_imputer.transform(train_df[numerical_columns])

            cat_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
            cat_imputer = cat_imputer.fit(test_df[categorical_columns])
            test_df[categorical_columns] = cat_imputer.transform(test_df[categorical_columns])

            num_imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            num_imputer = num_imputer.fit(test_df[numerical_columns])
            test_df[numerical_columns] = num_imputer.transform(test_df[numerical_columns])
            
            train_df["Date"]=train_df['Order_Date'].apply(lambda x:x.split("-")[0]).astype(int)
            train_df["Month"]=train_df['Order_Date'].apply(lambda x:x.split("-")[1]).astype(int)
            train_df['Ordered_Hour']=train_df['Time_Orderd'].str.split(':').str[0].astype(float)
            train_df['Ordered_Min']=train_df['Time_Orderd'].str.split(':').str[1].astype(float)
            train_df['Order_pick_Hour']=train_df['Time_Order_picked'].str.split(':').str[0].astype(float)
            train_df['Order_pick_Min']=train_df['Time_Order_picked'].str.split(':').str[1].astype(float)
            

            test_df["Date"]=test_df['Order_Date'].apply(lambda x:x.split("-")[0]).astype(int)
            test_df["Month"]=test_df['Order_Date'].apply(lambda x:x.split("-")[1]).astype(int)
            test_df['Ordered_Hour']=test_df['Time_Orderd'].str.split(':').str[0].astype(float)
            test_df['Ordered_Min']=test_df['Time_Orderd'].str.split(':').str[1].astype(float)
            test_df['Order_pick_Hour']=test_df['Time_Order_picked'].str.split(':').str[0].astype(float)
            test_df['Order_pick_Min']=test_df['Time_Order_picked'].str.split(':').str[1].astype(float)
            

            mean_time_train = train_df.groupby('Delivery_person_ID')['Time_taken (min)'].mean().sort_values()
            train_df['Delivery_person_ID'] = train_df['Delivery_person_ID'].apply(lambda x: mean_time_train[x])

            mean_time_test = test_df.groupby('Delivery_person_ID')['Time_taken (min)'].mean().sort_values()
            test_df['Delivery_person_ID'] = test_df['Delivery_person_ID'].apply(lambda x: mean_time_test[x])

            train_df["Ordered_Min"] = train_df["Ordered_Min"].fillna(train_df["Ordered_Min"].median())
            train_df["Order_pick_Min"] = train_df["Order_pick_Min"].fillna(train_df["Order_pick_Min"].median())

            test_df["Ordered_Min"] = test_df["Ordered_Min"].fillna(test_df["Ordered_Min"].median())
            test_df["Order_pick_Min"] = test_df["Order_pick_Min"].fillna(test_df["Order_pick_Min"].median())

            #clus = train_df.loc[:,["Restaurant_latitude","Restaurant_longitude"]]
            #clus
            #kmeans = KMeans(n_clusters = 4, init ='k-means++')
            #kmeans.fit(clus[clus.columns])
            #train_df['Restaurant_cluster_label'] = kmeans.predict(clus[clus.columns])

            #d_clus = train_df.loc[:,["Delivery_location_latitude","Delivery_location_longitude"]]
            #d_clus
            #kmeans = KMeans(n_clusters = 4, init ='k-means++')
            #kmeans.fit(d_clus[d_clus.columns])
            #train_df['Delivery_cluster_label'] = kmeans.predict(d_clus[d_clus.columns])


            #clus = test_df.loc[:,["Restaurant_latitude","Restaurant_longitude"]]
            #clus
            #kmeans = KMeans(n_clusters = 4, init ='k-means++')
            #kmeans.fit(clus[clus.columns])
            #test_df['Restaurant_cluster_label'] = kmeans.predict(clus[clus.columns])

            #d_clus = test_df.loc[:,["Delivery_location_latitude","Delivery_location_longitude"]]
            #d_clus
            #kmeans = KMeans(n_clusters = 4, init ='k-means++')
            #kmeans.fit(d_clus[d_clus.columns])
            #test_df['Delivery_cluster_label'] = kmeans.predict(d_clus[d_clus.columns])
            

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Time_taken (min)"
            drop_columns = [target_column_name,"ID","Order_Date","Time_Orderd","Time_Order_picked",
                            "Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude",
                            "Delivery_location_longitude"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]


            ## Transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            
            raise CustomException(e,sys)
            



        
