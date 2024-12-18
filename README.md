
# Digit Recognition

This is a python source code to predict digit. With my code, you can run the application to recognition digit through file png or jpe

## Dataset

Modified National Institute of Standards and Technology (MNIST)
![MnistExamples.png](MnistExamples.png)
## Trained model: LeNet
You could find my trained model at `lenet_model.pt`
## Run Locally

Clone the project

```bash
    git clone https://github.com/khoapdaio/digit_recognition.git
```

Go to the project directory

```bash
    cd digit_recognition
```

Create new enviroment
```bash
    python3 -m venv env
```

Activate new enviroment
```bash
    source env/bin/activate
```

Install dependencies

```bash
    pip install -r requirements.txt
```

Start app

```bash
    streamlit run app.py
```


## License

[![GitHub license](https://img.shields.io/github/license/khoapdaio/digit_recognition)](https://github.com/khoapdaio/quick-draw/blob/main/LICENSE)
