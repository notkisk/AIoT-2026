#include <Arduino.h>
#include <DHT.h>
#include <math.h>

#define DHTPIN 2
#define DHTTYPE DHT11


/*
 *we had these values for our logistic regression model trained to predict fire
 using two features for simplicity(Temperature and Humidity)
    Test accuracy: 0.7711160785566022
    MEAN = [15.953115080632283, 48.57430185214753]
    STD = [14.358038477662703, 8.82118774162391]
    WEIGHTS = [-0.2086760553936151, 0.9679781403658624]
    BIAS = 0.9693804113331885
 * 
 * */

class LogisticRegression {
private:
    static constexpr int N_FEATURES = 2;

    double mean[N_FEATURES];
    double std[N_FEATURES];
    double weights[N_FEATURES];
    double bias;

    float standardize(float x_raw, int idx) const {
        return (x_raw - mean[idx]) / std[idx];
    }

    float sigmoid(float z) const {
        return 1.0f / (1.0f + expf(-z));
    }

public:
    LogisticRegression(
        const double mean_[N_FEATURES],
        const double std_[N_FEATURES],
        const double weights_[N_FEATURES],
        double bias_
    ) : bias(bias_) {
        for (int i = 0; i < N_FEATURES; i++) {
            mean[i] = mean_[i];
            std[i] = std_[i];
            weights[i] = weights_[i];
        }
    }

    float predict(const float raw_features[N_FEATURES]) const {
        float z = 0.0f;

        for (int i = 0; i < N_FEATURES; i++) {
            float x = standardize(raw_features[i], i);
            z += weights[i] * x;
        }

        return sigmoid(z + bias);
    }

    int predict_class(const float raw_features[N_FEATURES]) const {
        return predict(raw_features) >= 0.5f ? 1 : 0;
    }
};

constexpr int N_FEATURES = 2;
constexpr double MEAN[N_FEATURES] = {15.95311508, 48.5743018};
constexpr double STD[N_FEATURES] = {14.3580384, 8.82118774};
constexpr double WEIGHTS[N_FEATURES] = {-0.2086760553936151, 0.9679781403658624};
constexpr double BIAS = 0.969380411331885;

LogisticRegression model(MEAN, STD, WEIGHTS, BIAS);
DHT dht(DHTPIN, DHTTYPE);

void setup() {
    Serial.begin(9600);
    dht.begin();
}

void loop() {
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();

    if (isnan(temp) || isnan(hum)) {
        Serial.println("DHT11 read failed!");
        delay(2000);
        return;
    }

    float features[N_FEATURES] = {temp, hum};

    float y_pred = model.predict(features);

    Serial.print("Temperature: ");
    Serial.print(temp);
    Serial.print(" C | Humidity: ");
    Serial.print(hum);
    Serial.print(" % | Probability: ");
    Serial.println(y_pred, 4);

    Serial.print("Predicted Class: ");
    Serial.println(model.predict_class(features));

    delay(2000);
}
