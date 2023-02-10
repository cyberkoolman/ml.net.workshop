using Microsoft.ML;

string dataFile = "corefx-issues-train.tsv";
string ModelPath = "GitHubLabeler.zip";

var mlContext = new MLContext();

// STEP 1: Common data loading configuration
var trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(
                        path: dataFile, 
                        hasHeader: true, 
                        separatorChar:'\t', 
                        allowSparse: false);
  
// STEP 2: Common data process configuration with pipeline data transformations
var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName:nameof(GitHubIssue.Area), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName:nameof(GitHubIssue.Title), outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(GitHubIssue.Description), outputColumnName: "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate(outputColumnName:"Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

// STEP 3: Create a training algorithm/trainer
var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");

// STEP 4: Create a training pipeline
// Need to map label to value (back to original readable state)
var trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// STEP 5: Train the model fitting to the dataset
Console.WriteLine("=============== Training the model ===============");
var trainedModel = trainingPipeline.Fit(trainingDataView);
Console.WriteLine("=============== Training finished ===============");

// STEP 6: Create prediction engine related to the loaded trained model
var predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(trainedModel);

// (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
var issue = new GitHubIssue() {
   ID = "Any-ID",
   Title = "WebSockets communication is slow in my machine",
   Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." 
};

var prediction = predEngine.Predict(issue);
Console.WriteLine($"=============== Single Prediction ===============");
Console.WriteLine($"Title: {issue.Title}");
Console.WriteLine($"Description: {issue.Description}");
Console.WriteLine($"Prediction Result: {prediction.Area}");

// STEP 7: Save/persist the trained model to a .ZIP file
Console.WriteLine("=============== Saving the model to a file ===============");
mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);