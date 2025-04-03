
使用指南

1、安装依赖
pip install camel-ai  -U

2、设置调用openai的api环境变量

windows
$env:OPENAI_API_KEY=""
macos和linux
export OPENAI_API_KEY=""


3、

1、生成示例数据：
python main.py --generate_sample_csv 
2、分析指定csv文件的数据：
python main.py --csv_file ObesityDataSet.csv

3、分析指定csv文件的数据的指定行：
python main.py --csv_file data.csv --row_index 5     