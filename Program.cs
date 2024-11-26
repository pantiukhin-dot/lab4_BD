using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

class Program
{
    // Structure for data from the CSV file
    public class TaxiData
    {
        [LoadColumn(1)] // Load timestamp (index 1 is for the 'timestamp' column)
        public string Timestamp { get; set; }

        [LoadColumn(2)] // Load value (index 2 is for the 'value' column)
        public float Value { get; set; }
    }

    // Structure for anomaly prediction results
    public class AnomalyPrediction
    {
        public float Score { get; set; } // Anomaly score
    }

    static void Main(string[] args)
    {
        // Create an instance of MLContext (entry point for ML.NET)
        var context = new MLContext();

        // Path to the data file
        string taxiDataPath = @"C:\Users\dataset.csv";  // Replace with the actual path to the CSV file

        // Load CSV data
        var taxiData = LoadCsvData(context, taxiDataPath);

        // Split the data into training and testing sets (80% for training, 20% for testing)
        var trainTestSplit = context.Data.TrainTestSplit(context.Data.LoadFromEnumerable(taxiData), testFraction: 0.2);

        // Create a data processing pipeline
        var pipeline = context.Transforms.Concatenate("Features", nameof(TaxiData.Value))  // Concatenate 'Value' into 'Features'
                        .Append(context.Transforms.NormalizeMeanVariance("Features"))  // Normalize the features
                        .Append(context.AnomalyDetection.Trainers.RandomizedPca(featureColumnName: "Features", rank: 1)); // Use PCA for anomaly detection

        // Train the model on the training set
        var model = pipeline.Fit(trainTestSplit.TrainSet);

        // Save the trained model to a file
        string modelPath = @"C:\Users\model.zip";
        context.Model.Save(model, trainTestSplit.TrainSet.Schema, modelPath);
        Console.WriteLine($"Model saved");

        // Load the model from the file
        ITransformer loadedModel = context.Model.Load(modelPath, out var modelInputSchema);
        Console.WriteLine("Model loaded from file");

        // Make predictions on the test set
        var predictions = loadedModel.Transform(trainTestSplit.TestSet);

        // Output the anomaly scores for the predictions
        var predictionResults = context.Data.CreateEnumerable<AnomalyPrediction>(predictions, reuseRowObject: false).ToList();
        foreach (var result in predictionResults)
        {
            Console.WriteLine($"Anomaly Score: {result.Score}");
        }

        // Create a sample data point for prediction
        var sampleData = new TaxiData
        {
            Timestamp = "2024-11-26 12:00:00",
            Value = 15000 // Example value for prediction
        };

        // Create a prediction engine and predict the anomaly score for the new sample data
        var predictionEngine = context.Model.CreatePredictionEngine<TaxiData, AnomalyPrediction>(loadedModel);
        var prediction = predictionEngine.Predict(sampleData);
        Console.WriteLine($"Anomaly Score: {prediction.Score}");

        // Wait for the user to press Enter to exit
        Console.WriteLine("Press Enter to exit...");
        Console.ReadLine();
    }

    // Load CSV data from the file
    static List<TaxiData> LoadCsvData(MLContext context, string filePath)
    {
        // Load data from CSV file
        var data = context.Data.LoadFromTextFile<TaxiData>(filePath, separatorChar: ',', hasHeader: true);

        // Convert IDataView to a list of TaxiData objects
        var dataList = context.Data.CreateEnumerable<TaxiData>(data, reuseRowObject: false).ToList();

        return dataList;
    }
}
