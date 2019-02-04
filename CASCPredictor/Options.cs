using CommandLine;

namespace CASCPredictor
{
    public class Options
    {
        #region Global

        [Option(Required = true, HelpText = "0 = Sample, 1 = Train")]
        public Mode Mode { get; set; }

        [Option(Required = true, HelpText = "Model name prefix")]
        public string ModelPrefix { get; set; }

        [Option(Required = true, HelpText = "Path to the Training Data")]
        public string DataPath { get; set; }

        [Option(HelpText = "Sample character length. Defaults to 100")]
        public int SampleSize { get; set; } = 100;

        #endregion

        #region Training Specific

        [Option(SetName = "train", HelpText = "Amount of times to train. Defaults to 50")]
        public int Epochs { get; set; } = 50;

        [Option(SetName = "train", HelpText = "Amount of minibatches between each sample. Defaults to 1000")]
        public int SampleFrequency { get; set; } = 1000;

        #endregion

        #region Sampling Specific

        [Option(SetName = "sample", HelpText = "Amount of samples to produce")]
        public uint SampleCount { get; set; } = 100;

        [Option(SetName = "sample", HelpText = "Sample prefix. Randomly selected if not provided", Default = "")]
        public string SamplePrime { get; set; }

        #endregion
    }

    public enum Mode
    {
        Sample,
        Train
    }
}
