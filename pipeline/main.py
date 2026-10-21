from risk import risk_model
from short_interest_alpha import alpha_model

model1=risk_model()
print(model1.total)
model1.add_number(5)
print(model1.total)

model2=alpha_model()
print(model2.alpha)
model2.add_alpha("A")
print(model2.alpha)
