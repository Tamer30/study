from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
example_train['ordinal'] = ordinal_encoder.fit_transform(example_train)
print(example_train) 
