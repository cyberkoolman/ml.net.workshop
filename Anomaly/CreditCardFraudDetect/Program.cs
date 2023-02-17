using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

MLContext mlContext = new MLContext();

var allData = mlContext.Data.LoadFromTextFile<TransactionObservation>("creditcard.csv", separatorChar: ',', hasHeader: true);
var trainTestSplit = mlContext.Data.TrainTestSplit(allData, testFraction: 0.2);

IDataView trainDataView = trainTestSplit.TrainSet;
IDataView testDataView = trainTestSplit.TestSet;

Console.WriteLine("Show 4 fraud transactions (true)");
ShowObservationsFilteredByLabel(mlContext, testDataView, label: true, count: 4);

Console.WriteLine("Show 4 NOT-fraud transactions (false)");
ShowObservationsFilteredByLabel(mlContext, testDataView, label: false, count: 4);

string[] featureColumnNames = trainDataView.Schema.AsQueryable()
    .Select(column => column.Name)                               // Get all the column names
    .Where(name => name != nameof(TransactionObservation.Label)) // Do not include the Label column
    .Where(name => name != nameof(TransactionObservation.Time))  // Do not include the Time column. Not needed as feature column
  .ToArray();

IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.Concatenate("Features", featureColumnNames)
  .Append(mlContext.Transforms.DropColumns(new string[] { nameof(TransactionObservation.Time) }))
  .Append(mlContext.Transforms.NormalizeLpNorm(outputColumnName: "NormalizedFeatures", inputColumnName: "Features"));

// In Anomaly Detection, the learner assumes all training examples have label 0, as it only learns from normal examples.
// If any of the training examples has label 1, it is recommended to use a Filter transform to filter them out before training:
IDataView normalTrainDataView = mlContext.Data.FilterRowsByColumn(
                                input: trainDataView, 
                                columnName: nameof(TransactionObservation.Label), 
                                lowerBound: 0, 
                                upperBound: 1);

IEstimator<ITransformer> trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(
  options: new RandomizedPcaTrainer.Options
      {
          FeatureColumnName = "NormalizedFeatures",   // The name of the feature column. The column data must be a known-sized vector of Single.
          ExampleWeightColumnName = null,				      // The name of the example weight column (optional). To use the weight column, the column data must be of type Single.
          Rank = 28,									                // The number of components in the PCA.
          Oversampling = 20,							            // Oversampling parameter for randomized PCA training.
          EnsureZeroMean = true,						          // If enabled, data is centered to be zero mean.
          Seed = 1									                  // The seed for random number generation.
      });

EstimatorChain<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer);
Console.WriteLine("=============== Training model ===============");

TransformerChain<ITransformer> model = trainingPipeline.Fit(normalTrainDataView);
Console.WriteLine("=============== End of training process ===============");

Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
var predictions = model.Transform(testDataView);

AnomalyDetectionMetrics metrics = mlContext.AnomalyDetection.Evaluate(predictions);
Console.WriteLine($"************************************************************");
Console.WriteLine($"*       Metrics for RandomizedPca anomaly detection model      ");
Console.WriteLine($"*-----------------------------------------------------------");
Console.WriteLine($"*       Area Under ROC Curve:                       {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"*       Detection rate at false positive count: {metrics.DetectionRateAtFalsePositiveCount}");
Console.WriteLine($"************************************************************");


void ShowObservationsFilteredByLabel(MLContext mlContext, IDataView dataView, bool label = true, int count = 2)
{
    // Convert to an enumerable of user-defined type. 
    var data = mlContext.Data.CreateEnumerable<TransactionObservation>(dataView, reuseRowObject: false)
                                    .Where(x => Math.Abs(x.Label - (label ? 1 : 0)) < float.Epsilon)
                                    // Take a couple values as an array.
                                    .Take(count)
                                    .ToList();

    // Print to console
    data.ForEach(row => { row.PrintToConsole(); });
}
