defmodule EnsembleClusteringTest do
  use ExUnit.Case
  doctest EnsembleClustering

  require Logger

  test "iris accuricy of knn" do
    Logger.debug "iris accuricy of knn"
    iris = UCIDataLoader.load_iris
    sample = iris |> UCIDataLoader.sampling_with_replace(10)

    sample.data
    |> Enum.map(&({iris, &1} |> KNN.knn))
    |> Enum.map(&(&1 |> elem(1)))
    |> Enum.zip(sample.target_all_name)
    |> Enum.map(fn({l,r}) -> l == r end)
    |> IO.inspect
  end

  test "wine quality accuricy of knn" do
    Logger.debug "wine quality accuricy of knn"
    wine = UCIDataLoader.load_wine_quality
    sample = wine |> UCIDataLoader.sampling_with_replace(41)

    sample.data
    |> Enum.map(&({wine, &1} |> KNN.knn))
    |> Enum.map(&(&1 |> elem(1)))
    |> Enum.zip(sample.target_all_name)
    |> Enum.map(fn({l,r}) -> l == r end)
    |> IO.inspect
  end

  test "iris accuricy of Naive Bayes" do
    Logger.debug "iris accuricy of Naive Bayes"
    iris = UCIDataLoader.load_iris
    sample = iris |> UCIDataLoader.sampling_with_replace(10)

    sample.data
    |> Enum.map(&({iris, &1} |> NaiveBayes.naivebayes))
    |> Enum.map(&(&1 |> elem(1)))
    |> Enum.zip(sample.target_all_name)
    |> Enum.map(fn({l,r}) -> l == r end)
    |> IO.inspect
  end

  test "wine quality accuricy of Naive Bayes" do
    Logger.debug "wine quality accuricy of Naive Bayes"
    wine = UCIDataLoader.load_wine_quality
    sample = wine |> UCIDataLoader.sampling_with_replace(41)

    sample.data
    |> Enum.map(&({wine, &1} |> NaiveBayes.naivebayes))
    |> Enum.map(&(&1 |> elem(1)))
    |> Enum.zip(sample.target_all_name)
    |> Enum.map(fn({l,r}) -> l == r end)
    |> IO.inspect
  end

  test "test avg calculate" do
        [1,2,6],[5,3,4],[4,5,6],[5,6,7],
      ]
    length = 3
    amount = 4
    assert [(1+5+4+5)/4, (2+3+5+6)/4, (6+4+6+7)/4] == NaiveBayes.avg(lst, length, amount)
  end
  test "test sd calculate" do
    lst =
      [
        [
          [1,2,3],[2,3,4],[4,5,6],[5,6,7], # cluster name "1"
        ],
        [
          [4,2,6],[5,5,4],[5,13,6],[7,6,12], # cluster name "2"
        ],
      ]
    length = 3
    amount = 4
    each_amount = [4, 4]
    avgs = lst |> Enum.map(&(&1 |> NaiveBayes.avg(length, amount)))
    target_names = ["1", "2"]
    assert [[:math.sqrt(5/2)], []] == NaiveBayes.sd(lst, avgs, target_names, length, each_amount) |> IO.inspect
  end
end
