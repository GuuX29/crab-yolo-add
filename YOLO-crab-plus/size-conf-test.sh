for i in 640 1280 1920 2560 3200 3840 5120
do
  echo "Input size: $i"
  for conf in $(seq 0.1 0.1 0.9)
  do
    echo "Confidence: $conf"
    name="exptest-size-$i-conf-$conf"
    python detect-xml-2c-work.py --img $i --source ./crop --weights ./trained_weight/x-simam-20.pt --agnostic-nms --conf $conf --save-txt --name $name --nosave --project size-conf-test
  done
done
