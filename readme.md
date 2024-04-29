## 1. 数据整体探测
Untitled1.ipynb

## 2. 切分数据集和训练集
Untitled1.ipynb

## 3. 建议gpu环境下训练

~~~python
#训练
python garbage.py --model_name resnext101_32x16d --lr 0.001 --optimizer adam --start_epoch 1 --epochs 10 --num_classes 40

#评估
python garbage.py --model_name resnext101_32x16d --evaluate --resume checkpoint/checkpoint.pth.tar --num_classes 40

#预测
python app_garbage.py --model_name resnext101_32x16d --resume checkpoint/garbage_resnext101_model_2_7420_8430.pth
~~~

