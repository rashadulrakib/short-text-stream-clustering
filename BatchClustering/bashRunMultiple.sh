#!/bin/bash
for i in {400..400}
do
   echo "experiment $i"
   python main.py > eval-result/mstream-enhance/Tweets/mstream-out$i
   python evaluation.py > eval-result/mstream-enhance/Tweets/mstream-eval-out$i
done
