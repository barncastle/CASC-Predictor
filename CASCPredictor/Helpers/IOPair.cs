namespace CASCPredictor.Helpers
{
    /// <summary>
    /// Named and reusable 2 item Tuple
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal struct IOPair<T>
    {
        public T Input;
        public T Output;

        public IOPair(T input, T output)
        {
            Input = input;
            Output = output;
        }
    }
}
