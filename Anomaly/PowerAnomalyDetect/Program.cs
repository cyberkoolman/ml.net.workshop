using Microsoft.ML;
using Microsoft.ML.Data;

const string TrainingDataPath = "power-export_min.csv";
const string ModelPath = "power-anomaly.zip";

var mlContext = new MLContext();

// load data
var dataView = mlContext.Data.LoadFromTextFile<MeterData>(
    TrainingDataPath,
    separatorChar: ',',
    hasHeader: true);

// transform options
BuildTrainModel(mlContext, dataView);  // using SsaSpikeEstimator

DetectAnomalies(mlContext, dataView);

Console.WriteLine("\nPress any key to exit");
Console.Read();

static void BuildTrainModel(MLContext mlContext, IDataView dataView)
{
    // Configure the Estimator
    const int PValueHistoryLength = 30;
    const int SeasonalityWindowSize = 30;
    const int TrainingWindowSize = 90;
    const long ConfidenceInterval = 98L;            

    string outputColumnName = nameof(SpikePrediction.Prediction);
    string inputColumnName = nameof(MeterData.ConsumptionDiffNormalized);

    var trainigPipeLine = mlContext.Transforms.DetectSpikeBySsa(
        outputColumnName,
        inputColumnName,
        confidence: ConfidenceInterval,
        pvalueHistoryLength: PValueHistoryLength,
        trainingWindowSize: TrainingWindowSize,
        seasonalityWindowSize: SeasonalityWindowSize);

    ITransformer trainedModel = trainigPipeLine.Fit(dataView);

    // STEP 6: Save/persist the trained model to a .ZIP file
    mlContext.Model.Save(trainedModel, dataView.Schema, ModelPath);

    Console.WriteLine("The model is saved to {0}", ModelPath);
    Console.WriteLine("");
}

static void DetectAnomalies(MLContext mlContext,IDataView dataView)
{
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

    var transformedData = trainedModel.Transform(dataView);

    // Getting the data of the newly created column as an IEnumerable
    IEnumerable<SpikePrediction> predictions =
        mlContext.Data.CreateEnumerable<SpikePrediction>(transformedData, false);
    
    var colCDN = dataView.GetColumn<float>("ConsumptionDiffNormalized").ToArray();
    var colTime = dataView.GetColumn<DateTime>("time").ToArray();

    // Output the input data and predictions
    Console.WriteLine("======Displaying anomalies in the Power meter data=========");
    Console.WriteLine("Date              \tReadingDiff\tAlert\tScore\tP-Value");

    int i = 0;
    foreach (var p in predictions)
    {
        if (p.Prediction[0] == 1)
        {
            Console.BackgroundColor = ConsoleColor.DarkYellow;
            Console.ForegroundColor = ConsoleColor.Black;
        }
        Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}", 
            colTime[i], colCDN[i], 
            p.Prediction[0], p.Prediction[1], p.Prediction[2]);
        Console.ResetColor();
        i++;
    }
}
