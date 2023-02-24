using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

// This sample shows two different ML tasks and algorithms that can be used for forecasting:
// 1.) Regression using FastTreeTweedie Regression
// 2.) Time Series using Single Spectrum Analysis
// Each of these techniques are used to forecast monthly units for the same products
// so that you can compare the forecasts.

const string dataPath = "products.stats.csv";
int productId = 988;    // Forecast units sold for product with Id == 988
string outputModelPath = $"product{productId}_month_timeSeriesSSA.zip";

var mlContext = new MLContext(seed: 1);  //Seed set to any number so you have a deterministic environment

Console.WriteLine("Building the Forecast model using Time Series SSA estimation");

IDataView allProductsDataView = mlContext.Data.LoadFromTextFile<ProductData>(dataPath, hasHeader: true, separatorChar: ',');
IDataView productDataView = mlContext.Data.FilterRowsByColumn(allProductsDataView, nameof(ProductData.productId), productId, productId + 1);
var singleProductDataSeries = mlContext.Data.CreateEnumerable<ProductData>(productDataView, false).OrderBy(p => p.year).ThenBy(p => p.month);
ProductData lastMonthProductData = singleProductDataSeries.Last();

const int numSeriesDataPoints = 34; //The underlying data has a total of 34 months worth of data for each product

// Create and add the forecast estimator to the pipeline.
IEstimator<ITransformer> forecastEstimator = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: nameof(ProductUnitTimeSeriesPrediction.ForecastedProductUnits), 
    inputColumnName: nameof(ProductData.units), // This is the column being forecasted.
    windowSize: 12, // Window size is set to the time period represented in the product data cycle; our product cycle is based on 12 months, so this is set to a factor of 12, e.g. 3.
    seriesLength: numSeriesDataPoints, // This parameter specifies the number of data points that are used when performing a forecast.
    trainSize: numSeriesDataPoints, // This parameter specifies the total number of data points in the input time series, starting from the beginning.
    horizon: 2, // Indicates the number of values to forecast; 2 indicates that the next 2 months of product units will be forecasted.
    confidenceLevel: 0.95f, // Indicates the likelihood the real observed value will fall within the specified interval bounds.
    confidenceLowerBoundColumn: nameof(ProductUnitTimeSeriesPrediction.ConfidenceLowerBound), //This is the name of the column that will be used to store the lower interval bound for each forecasted value.
    confidenceUpperBoundColumn: nameof(ProductUnitTimeSeriesPrediction.ConfidenceUpperBound)); //This is the name of the column that will be used to store the upper interval bound for each forecasted value.

// Fit the forecasting model to the specified product's data series.
ITransformer forecastTransformer = forecastEstimator.Fit(productDataView);

// Create the forecast engine used for creating predictions.
TimeSeriesPredictionEngine<ProductData, ProductUnitTimeSeriesPrediction> tsForecastEngine
        = forecastTransformer.CreateTimeSeriesEngine<ProductData, ProductUnitTimeSeriesPrediction>(mlContext);

// Save the forecasting model so that it can be loaded within an end-user app.
tsForecastEngine.CheckPoint(mlContext, outputModelPath);
Console.WriteLine("Building and saving the model completed.");

ITransformer forecaster;
using (var file = File.OpenRead(outputModelPath))
{
    forecaster = mlContext.Model.Load(file, out DataViewSchema schema);
}

// We must create a new prediction engine from the persisted model.
TimeSeriesPredictionEngine<ProductData, ProductUnitTimeSeriesPrediction> predictEngine
         = forecaster.CreateTimeSeriesEngine<ProductData, ProductUnitTimeSeriesPrediction>(mlContext);

// Get the prediction; this will include the forecasted product units sold for the next 2 months since this the time period specified in the `horizon` parameter when the forecast estimator was originally created.
Console.WriteLine("\n** Original prediction **");
ProductUnitTimeSeriesPrediction originalSalesPrediction = predictEngine.Predict();

// Compare the units of the first forecasted month to the actual units sold for the next month.
var predictionMonth = lastMonthProductData.month == 12 ? 1 : lastMonthProductData.month + 1;
var predictionYear = predictionMonth < lastMonthProductData.month ? lastMonthProductData.year + 1 : lastMonthProductData.year;
Console.WriteLine($"Product: {lastMonthProductData.productId}," + 
                  $" Month: {predictionMonth}, Year: {predictionYear} " +  
                  $"- Real Value (units): {lastMonthProductData.next}, " +
                  $"Forecast Prediction (units): {originalSalesPrediction.ForecastedProductUnits[0]}");

// Get the first forecasted month's confidence interval bounds.
Console.WriteLine($"Confidence interval: [{originalSalesPrediction.ConfidenceLowerBound[0]} - {originalSalesPrediction.ConfidenceUpperBound[0]}]\n");

// Get the units of the second forecasted month.
Console.WriteLine($"Product: {lastMonthProductData.productId}, Month: {lastMonthProductData.month + 2}, Year: {lastMonthProductData.year}, " +
    $"Forecast (units): {originalSalesPrediction.ForecastedProductUnits[1]}");

// Get the second forecasted month's confidence interval bounds.
Console.WriteLine($"Confidence interval: [{originalSalesPrediction.ConfidenceLowerBound[1]} - {originalSalesPrediction.ConfidenceUpperBound[1]}]\n");

// Update the forecasting model with the next month's actual product data to get an updated prediction; this time, only forecast product sales for 1 month ahead.
Console.WriteLine("** Updated prediction **");
ProductUnitTimeSeriesPrediction updatedSalesPrediction = predictEngine.Predict(SampleProductData.MonthlyData[1], horizon: 1);

// Save a checkpoint of the forecasting model.
predictEngine.CheckPoint(mlContext, outputModelPath);

// Get the units of the updated forecast.
predictionMonth = lastMonthProductData.month >= 11 ? (lastMonthProductData.month + 2) % 12 : lastMonthProductData.month + 2;
predictionYear = predictionMonth < lastMonthProductData.month ? lastMonthProductData.year + 1 : lastMonthProductData.year;
Console.WriteLine($"Product: {lastMonthProductData.productId}, Month: {predictionMonth}, Year: {predictionYear}, " +
    $"Forecast (units): {updatedSalesPrediction.ForecastedProductUnits[0]}");

// Get the updated forecast's confidence interval bounds.
Console.WriteLine($"Confidence interval: [{updatedSalesPrediction.ConfidenceLowerBound[0]} - {updatedSalesPrediction.ConfidenceUpperBound[0]}]\n");
