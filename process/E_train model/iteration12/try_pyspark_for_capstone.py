from pyspark import SparkContext
from pyspark.shell import spark

VERSION = '11'

#sc = SparkContext()
#print(type(sc))
folder_prefix='../../../'

filename = f'df_listings_v{VERSION}.csv'

df_pathname_tidy = folder_prefix + f'data/final/{filename}'

sdf = spark.read\
                .options(header=True,inferSchem=True)\
                .csv(df_pathname_tidy)

#sdf = sdf.withColumnRenamed("","_c0")
sdf.show()

#try_agg = sdf.groupBy('bathrooms').pivot("bedrooms").sum()
#print(try_agg)

print(sdf.describe().show())

(train, test) = sdf.randomSplit([0.7, 0.3])

from pyspark.ml.feature import StringIndexer
#plan_indexer = StringIndexer(inputCol = 'Product_ID', outputCol = 'product_ID')
plan_indexer = StringIndexer(inputCol = 'tenureType', outputCol = 'tenureType2')
#labeller = plan_indexer.fit(train)

#Train1 = labeller.transform(train)
#Test1 = labeller.transform(test)
Train1 = train
Test1 = test

Train1.show()

from pyspark.ml.feature import RFormula
#formula = RFormula(formula="Purchase ~ Age+ Occupation +City_Category+Stay_In_Current_City_Years+Product_Category_1+Product_Category_2+ Gender",featuresCol="features",labelCol="label")
formula = RFormula(formula="Purchase ~ bedrooms",featuresCol="features",labelCol="Price")

t1 = formula.fit(Train1)
train1 = t1.transform(Train1)
test1 = t1.transform(Test1)

train1.show()

train1.select('features').show()
train1.select('label').show()

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor()

(train_cv, test_cv) = train1.randomSplit([0.7, 0.3])

model1 = rf.fit(train_cv)
predictions = model1.transform(test_cv)

model1 = rf.fit(train_cv)
predictions = model1.transform(test_cv)


from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
mse = evaluator.evaluate(predictions,{evaluator.metricName:"mse" })
import numpy as np
np.sqrt(mse), mse

model = rf.fit(train1)
predictions1 = model.transform(test1)

df = predictions1.selectExpr("User_ID as User_ID", "Product_ID as Product_ID", 'prediction as Purchase')

df.toPandas().to_csv('submission.csv')