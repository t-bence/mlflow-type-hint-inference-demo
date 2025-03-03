# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow signatures from type hints
# MAGIC
# MAGIC This notebook introduces a new feature of MLflow: that it can infer the schema of a `PythonFunction` model from its type hints.
# MAGIC
# MAGIC You can find some interesting resources here: https://www.mlflow.org/docs/latest/model/python_model.html#model-signature-inference-based-on-type-hints
# MAGIC
# MAGIC Preparations: Update mlflow to 2.20, which introduces type hint based signatures. Let's fix the pydantic version as well, this is needed to make the code work on DBRs 15.4 LTS ML and 16.2 ML
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow-skinny==2.20.* pydantic==2.10.6

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Let's define a simple model that predicts the probability of the tresure being at a certain location! The input will be a custom `Coordinate` object, and the output a probability, so a simple `float`.

# COMMAND ----------

from pydantic import BaseModel

class Coordinate(BaseModel):
    latitude: float
    longitude: float

# COMMAND ----------

# MAGIC %md
# MAGIC Let's define the custom model using `mlflow`'s `PythonModel` class! The model expects a bunch of inputs at a time, so a `list[Coordinate]`, and the output is a list of probabilities, so a `list[float]`

# COMMAND ----------

from mlflow.pyfunc import PythonModel

class TreasureModel(PythonModel):
    def predict(self, model_input: list[Coordinate]) -> list[float]:
        import random
        return [random.uniform(0, 1) for item in model_input]
    
treasure_model = TreasureModel()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create some sample data and check that it works as expected:

# COMMAND ----------

datapao_office = Coordinate(latitude=47.50357640074957, longitude=19.062161198478307)
middle_of_atlantic = Coordinate(latitude=-55.19020425256697, longitude=-11.699930054286687)

coordinates = [datapao_office, middle_of_atlantic]

for proba in treasure_model.predict(coordinates):
    print(proba)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's predict on some dummy data as well

# COMMAND ----------

treasure_model.predict([{"latitude": 1.0, "longitude": 1.0}])

# COMMAND ----------

# MAGIC %md 
# MAGIC Now let's try it on some data that has wrong data type -- here, `longitude` is a string instad of a float.
# MAGIC
# MAGIC Notice the nice descriptive error message!

# COMMAND ----------

try:
    treasure_model.predict([{"latitude": 1.0, "longitude": "apples"}])
except Exception as e:
    print("Cannot predict!")
    print(e)


# COMMAND ----------

# MAGIC %md
# MAGIC Another experiment: a field is completely missing -- again, we get an exception! Nice!

# COMMAND ----------


try:
    treasure_model.predict([{"latitude": 1.0}])
except Exception as e:
    print("Cannot predict!")
    print(e)


# COMMAND ----------

# MAGIC %md
# MAGIC Let's log the model to mlflow!

# COMMAND ----------

import mlflow

mlflow.end_run()

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        "treasure_model",
        python_model=treasure_model,
        input_example=coordinates
    )


# COMMAND ----------

# MAGIC %md
# MAGIC We can load it back, and it will still work!

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"runs:/{model_info.run_id}/treasure_model")

model.predict([{"latitude": 1.0, "longitude": 1.0}])

# COMMAND ----------

# MAGIC %md
# MAGIC We can inspect the stored schema:

# COMMAND ----------

print("Input:")
print(model.metadata.get_input_schema())
print("Output:")
print(model.metadata.get_output_schema())

# COMMAND ----------

# MAGIC %md
# MAGIC On Databricks, we could also register the model into Unity Catalog. It picks up the schema as well, but only shows the input type as "object" and the output as "float"

# COMMAND ----------

# mlflow.set_registry_uri("databricks-uc")
# model_name = "bence_toth.testing.dummy-model"
# reg_info = mlflow.register_model(f"runs:/{model_info.run_id}/treasure_model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC We can load the models back as UDFs and use them for inference

# COMMAND ----------

treasure_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"runs:/{model_info.run_id}/treasure_model")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a simple dataframe and run the UDF on it

# COMMAND ----------

import pyspark.sql.functions as F

spark_df = (spark.createDataFrame([(c,) for c in coordinates], schema=["coordinates"])
    .withColumn("treasure", treasure_udf("coordinates"))
)

spark_df.display()
