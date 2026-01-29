for i in {1..500}
do
  echo "Run $i"
  python create_artificial_runs.py &
done
wait