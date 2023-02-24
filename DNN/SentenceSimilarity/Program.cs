using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MathNet.Numerics.Statistics;
using Microsoft.ML.Transforms;

const string modelName = "model.zip";

// Initialize MLContext
var ctx = new MLContext();

// (Optional) Use GPU
ctx.GpuDeviceId = 0;
ctx.FallbackToCpu = true;

// Log training output
ctx.Log += (o, e) => {
    if (e.Source.Contains("NasBertTrainer"))
        Console.WriteLine(e.Message);
};

if (args.Length > 0) {
  if (args[0].ToLower() == "inference") {
    var ssModel = ctx.Model.Load(modelName, out var _);

    var sampleData = new ModelInput() {
        Product = "Single-Handle Pull-Down Sprayer",
        SearchTerm = "Computer"
    };

    PredictionEngine<ModelInput, ModelOutput> engine = ctx.Model.CreatePredictionEngine<ModelInput, ModelOutput>(ssModel);
    ModelOutput result = engine.Predict(sampleData);
    Console.WriteLine($"Outcome: {result.Score}");  
    return;
  }
} 

IDataView dataView = ctx.Data.LoadFromTextFile<ModelInput>(
    "home-depot-products.csv",
    hasHeader: true,
    separatorChar: ','
);

// Split data into 80% training, 20% testing
var dataSplit = ctx.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;

// Define pipeline
var featurePipeline = ctx.Transforms.ReplaceMissingValues("Label", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);
var trainer = ctx.Regression.Trainers.SentenceSimilarity(
                labelColumnName: "Label",
                sentence1ColumnName: "Product", 
                sentence2ColumnName: "SearchTerm");
var pipeline = featurePipeline.Append(trainer);

// Train the model
var model = pipeline.Fit(dataSplit.TrainSet);

// Save the model
ctx.Model.Save(model, trainData.Schema, modelName);

// Evaluate the model
var predictions = model.Transform(testData);
var actual = predictions.GetColumn<float>("Label")
            .Select(x => (double)x);
var predicted = predictions.GetColumn<float>("Score")
            .Select(x => (double)x);
var corr = Correlation.Pearson(actual, predicted);
Console.WriteLine($"Pearson Correlation: {corr}");
