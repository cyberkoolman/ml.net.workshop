using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

const string ModelName = "subjects.zip";

MLContext mlContext = new MLContext()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};
IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    "subjects-questions.tsv",
    hasHeader: true,
    separatorChar: '\t'
);
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;

// Create a pipeline for training the model
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "Label",
        inputColumnName: "Label")
    .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
        labelColumnName: "Label", 
        sentence1ColumnName: "Content",
        architecture: BertArchitecture.Roberta))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
        outputColumnName: "PredictedLabel", 
        inputColumnName: "PredictedLabel"));

Console.WriteLine("Building model and wait until completion ......");
ITransformer tcmodel = pipeline.Fit(trainData);
mlContext.Model.Save(tcmodel, trainData.Schema, ModelName);
Console.WriteLine($"Done. Saved model to model {ModelName}");
