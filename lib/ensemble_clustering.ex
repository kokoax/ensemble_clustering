defmodule KNN do
  def knn({datasets, data}) do
    Enum.zip(datasets.data, datasets.target_all_name)
      |> Enum.map(&([dist(data, &1 |> elem(0)), &1 |> elem(1)]))
      |> Enum.sort
      |> vote(3,datasets.target_names)
  end
  defp vote(sorted, k, target_names) do
    vote_names = 0..k-1
      |> Enum.map(&(sorted |> Enum.at(&1) |> Enum.at(1)))
      # |> Enum.map(&(counter = counter |> Map.update(&1,1,fn x -> x+1 end)))
    target_names
      |> Enum.map(
        fn(tname) ->
          {
            (vote_names
              |> Enum.filter(&(&1 == tname))
              |> Enum.count), tname
          }
        end)
      |> Enum.max # TODO: More check detaily
  end
  defp dist(x, y) do
    # IO.inspect x
    # IO.inspect y
    Enum.zip(x,y)
      |> Enum.map(&((&1 |> elem(0)) - (&1 |> elem(1))))
      |> Enum.map(&(&1 |> :math.pow(2)))
      |> Enum.sum
      |> :math.sqrt
  end
end

defmodule NaiveBayes do
 def naivebayes({datasets,data}) do
    data_with_names = Enum.zip(datasets.data, datasets.target_all_name)
    avg =
      Enum.zip(datasets.target_names, datasets.each_amount)
        |> Enum.map(
          fn(name_with_amount) ->
            data_with_names
              |> Enum.filter(&(&1 |> elem(1) == name_with_amount |> elem(0)))
              |> Enum.map(&(&1 |> elem(0)))
              |> avg(datasets.length,name_with_amount |> elem(1)) # TODO: to generalize
          end
        )
      sd =
        datasets.target_names
          |> Enum.map(
            fn(name) ->
              data_with_names
                |> Enum.filter(&(&1 |> elem(1) == name))
                |> Enum.map(&(&1 |> elem(0)))
            end
          )
          |> sd(avg,datasets.target_names,datasets.length,datasets.each_amount) # TODO: to generalize
    Enum.zip(avg,sd)
      |> Enum.map(
        fn(avg_with_sd) ->
          avg = avg_with_sd |> elem(0)
          sd  = avg_with_sd |> elem(1)
          Enum.zip(avg,sd) |> Enum.zip(data)
            |> Enum.map(
              fn(zipped) ->
                tmp  = zipped |> elem(0)
                avg  =    tmp |> elem(0)
                sd   =  (tmp |> elem(1)) + 1
                data = zipped |> elem(1)
                (1/:math.sqrt(2*:math.pi*:math.pow(sd,2))) * :math.exp(-1*:math.pow(data-avg,2) / (2*:math.pow(sd,2)))
                # p[i] = (1 / sqrt(2*M_PI*pow(sd,2))) * exp(-1 * pow(target->num[i]-avg,2) / (2*pow(sd,2)))
              end
            )
            |> Enum.sum
        end
      )
      |> Enum.zip(datasets.target_names)
      # |> Enum.zip(datasets.target_names)
      |> Enum.max
  end
  # Iris行列のそれぞれの特徴ごと(列)の平均を計算
  def avg(lst, length, amount) do
    0..length-1
    |> Enum.map(
      fn(iter) ->
        lst
          # Iris matrix transpose
          |> Enum.map(&(&1 |> Enum.at(iter)))
          |> Enum.sum
      end
    )
    |> Enum.map(&(&1 / amount))
  end
  # Iris行列のそれぞれの特徴ごと(列)の標準偏差を計算
  def sd(lst, avgs, target_names, length, each_amount) do
    Enum.zip(avgs,lst)
      |> Enum.map(
        fn(avg_with_data) ->
          avg = avg_with_data |> elem(0)
          features = avg_with_data |> elem(1)
          features |>
            Enum.map(
              fn(feature) ->
                0..length-1
                  |> Enum.map(
                    fn(iter) ->
                      :math.pow((feature |> Enum.at(iter)) - (avg |> Enum.at(iter)), 2)
                    end
                  )
              end
            )
        end
      )
      |> Enum.map(
        fn(var_on_cluster) ->
            Enum.zip(0..length-1, each_amount)
              |> Enum.map(
                fn(column_with_amount) ->
                  var_on_cluster |>
                    Enum.map(
                      fn(var_on_features) ->
                        var_on_features
                          |> Enum.at(column_with_amount |> elem(0))
                      end
                    )
                    |> Enum.sum
                    |> frac(column_with_amount |> elem(1))
                    |> :math.sqrt
                end
              )
              # |> Enum.map(&((&1 |> Enum.sum)/(column_with_amount |> elem(1))|> :math.sqrt))
        end
      )
  end
  defp frac(x, y) do
    x / y
  end
end

defmodule ID3 do
  def id3({datasets,data}) do
    # IO.inspect datasets
    # IO.inspect data
  end
end

defmodule EnsembleClustering do
  # iris set
  # @sample   30
  # @test_num 10

  # wine quality set
  @sample 320
  @test_num 41
  def test(datasets) do
    datasets
      |> get_test_data(@test_num)
      |> Enum.map(
        fn(test_dataset) ->
          test_dataset
            |> Enum.map(
              fn(test_data) ->
                datasets
                  |> bagging(test_data)
              end
            )
            |> Enum.filter(&(&1 == true))
            |> Enum.count
            |> frac(@test_num)
        end
      )
      |> Enum.sum
      |> frac(datasets.amount/@test_num)
  end
  defp frac(x, y) do
    x / y
  end
  def get_test_data(datasets, n) do
    0..round(datasets.amount/n-1)
      |> Enum.map(
        fn(i) ->
          (i*n)..(i*n+n-1)
            |> Enum.map(&(
              {datasets.data |> Enum.at(&1), datasets.target_all_name |> Enum.at(&1)}
            ))
        end
      )
  end
  def bagging(datasets, data) do
    knn_results = 1..10
      |> Enum.map(
        fn(_) ->
          {UCIDataLoader.sampling_with_replace(datasets,@sample),data |> elem(0)}
            |> KNN.knn
        end
      )

    nb_results  = 1..10
      |> Enum.map(
        fn(_) ->
          {UCIDataLoader.sampling_with_replace(datasets,@sample),data |> elem(0)}
            |> NaiveBayes.naivebayes
        end
      )

    datasets.target_names
    |> Enum.map(
      fn(tname) ->
        {
          ((knn_results |> Enum.map(&(&1 |> elem(1)))) ++ (nb_results  |> Enum.map(&(&1 |> elem(1))))
            |> Enum.filter(
              fn(vote_name) ->
                tname == vote_name
              end
            ) |> Enum.count),
          tname
        }
      end
    )
    |> Enum.max |> elem(1) == data |> elem(1)
    # {datasets,data} |> ID3.id3
  end
end

defmodule Main do
  def main(datasets) do
    datasets |> EnsembleClustering.test
    # wine = UCIDataLoader.load_wine_quality
    # wine |> EnsembleClustering.test
  end
end

