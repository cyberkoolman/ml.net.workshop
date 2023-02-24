using Microsoft.ML;
using Microsoft.ML.Trainers;

const string TrainingDataLocation = "recommendation-ratings-train.csv";
const string TestDataLocation = "recommendation-ratings-test.csv";

const float predictionuserId = 6;
const int predictionmovieId = 189333;   // Mission: Impossible - Fallout (2018)

MLContext mlcontext = new MLContext();

//STEP 1: Read the training data which will be used to train the movie recommendation model    
//The schema for training data is defined by type 'TInput' in LoadFromTextFile<TInput>() method.
IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TrainingDataLocation, hasHeader: true, separatorChar:',');

//STEP 2: Transform your data by encoding the two features userId and movieID. 
//        These encoded features will be provided as input
//        to our MatrixFactorizationTrainer.
var dataProcessingPipeline = mlcontext.Transforms.Conversion.MapValueToKey(
                              outputColumnName: "userIdEncoded", inputColumnName: nameof(MovieRating.userId))
                            .Append(mlcontext.Transforms.Conversion.MapValueToKey(
                              outputColumnName: "movieIdEncoded", inputColumnName: nameof(MovieRating.movieId)));

//Specify the options for MatrixFactorization trainer            
MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
options.MatrixColumnIndexColumnName = "userIdEncoded";
options.MatrixRowIndexColumnName = "movieIdEncoded";
options.LabelColumnName = "Label";
options.NumberOfIterations = 20;
// The approximation rank specifies the number of latent factors used in the factorization. 
// Each latent factor represents a hidden feature that characterizes the user or the item.
options.ApproximationRank = 100;

//STEP 3: Create the training pipeline 
var trainingPipeLine = dataProcessingPipeline.Append(
                        mlcontext.Recommendation().Trainers.MatrixFactorization(options));

//STEP 4: Train the model fitting to the DataSet
Console.WriteLine("=============== Training the model ===============");
ITransformer model = trainingPipeLine.Fit(trainingDataView);

//STEP 5: Evaluate the model performance 
Console.WriteLine("=============== Evaluating the model ===============");
IDataView testDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true, separatorChar: ','); 
var prediction = model.Transform(testDataView);
var metrics = mlcontext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);

//STEP 6:  Try/test a single prediction by predicting a single movie rating for a specific user
Console.WriteLine("=============== Making the sample prediction ===============");
var predictionengine = mlcontext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
/* Make a single movie rating prediction, the scores are for a particular user and will range from 1 - 5. 
    The higher the score the higher the likelyhood of a user liking a particular movie.
    You can recommend a movie to a user if say rating > 3.5.*/
var movieratingprediction = predictionengine.Predict(
    new MovieRating()
    {
        //Example rating prediction for userId = 6, movieId = 189333
        userId = predictionuserId,
        movieId = predictionmovieId
    }
);

Movie movieService = new Movie();
Console.WriteLine($"For userId: {predictionuserId} "
      + $" movie rating prediction (1 - 5 stars) for movie: {movieService.Get(predictionmovieId).movieTitle}"
      + $" is: {Math.Round(movieratingprediction.Score, 1)}");
