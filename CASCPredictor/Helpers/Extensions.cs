using System;
using System.Collections.Generic;

namespace CASCPredictor.Helpers
{
    internal static class Extensions
    {
        /// <summary>
        /// Randomly shuffles an array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <returns></returns>
        public static T[] Shuffle<T>(this T[] array)
        {
            var rnd = new Random();

            int n = array.Length;
            while (n > 1)
            {
                int k = rnd.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }

            return array;
        }

        /// <summary>
        /// Picks a random element from an array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <returns></returns>
        public static T PickRandom<T>(this T[] array)
        {
            if (array.Length == 0)
                return default(T);
            else if (array.Length == 1)
                return array[0];
            else
                return array[new Random().Next(array.Length)];
        }

        /// <summary>
        /// Sugar syntax for iterating an enumerable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="enumerable"></param>
        /// <param name="action"></param>
        public static void ForEach<T>(this IEnumerable<T> enumerable, Action<T> action)
        {
            var enumerator = enumerable.GetEnumerator();
            while (enumerator.MoveNext())
                action(enumerator.Current);
        }

        /// <summary>
        /// Dequeue multiple values from a Queue
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="queue"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public static IEnumerable<T> DequeueRange<T>(this Queue<T> queue, int count)
        {
            for (int i = 0; i < count && queue.Count > 0; i++)
                yield return queue.Dequeue();
        }
    }
}
