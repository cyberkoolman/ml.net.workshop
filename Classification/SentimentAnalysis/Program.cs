using Microsoft.ML;

bool IsTrainingMode = false;
string DataPath = "wikiDetoxAnnotated40kRows.tsv";
string ModelPath = "SentimentModel.zip";

var mlContext = new MLContext();

if (IsTrainingMode) {
    // STEP 1: Common data loading configuration
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

    var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    IDataView trainingData = trainTestSplit.TrainSet;
    IDataView testData = trainTestSplit.TestSet;

    // STEP 2: Common data process configuration with pipeline data transformations          
    var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                            outputColumnName: "Features", 
                            inputColumnName: nameof(SentimentIssue.Text));

    // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
    var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
          labelColumnName: "Label", 
          featureColumnName: "Features");
    var trainingPipeline = dataProcessPipeline.Append(trainer);

    // STEP 4: Train the model fitting to the DataSet
    var trainedModel = trainingPipeline.Fit(trainingData);

    // STEP 5: Evaluate the model and show accuracy stats
    var predictions = trainedModel.Transform(testData);
    var metrics = mlContext.BinaryClassification.Evaluate(
          data: predictions, 
          labelColumnName: "Label", 
          scoreColumnName: "Score");

    Console.WriteLine($"Accuracy: {metrics.Accuracy}");
    Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
    Console.WriteLine($"F1 Score: {metrics.F1Score}");
    Console.WriteLine($"Log Loss: {metrics.LogLoss}");
    Console.WriteLine();

    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

    mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);
    Console.WriteLine("The model is saved to {0}", ModelPath);
} else {
    // TRY IT: Make a single test prediction, loading the model from .ZIP file
    DataViewSchema modelSchema;
    var sentimentModel = mlContext.Model.Load(ModelPath, out modelSchema);

    SentimentIssue sampleStatement = new SentimentIssue { Text = "Your service is crappy *0(*!!!!!!s****." };
    // SentimentIssue sampleStatement = new SentimentIssue { Text = "Not the best, imo" };

    // Create prediction engine related to the loaded trained model
    var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(sentimentModel);

    // Score
    var resultprediction = predEngine.Predict(sampleStatement);

    Console.WriteLine($"=============== Single Prediction  ===============");
    Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
    Console.WriteLine($"================End of Process.Hit any key to exit==================================");
    Console.ReadLine();
}
