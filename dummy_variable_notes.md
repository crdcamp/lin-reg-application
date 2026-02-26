# The Dummy Variable Trap

[Resource](https://www.learndatasci.com/glossary/dummy-variable-trap/)

The dummy variable trap occurs when two or more dummy variables created by one-hot encoding are highly correlated (multi-collinear). This means that one variable can be predicted from the others, making it difficult to interpret predicted coefficient variables in regression models.

In other words, the individual effect of the dummy variables on the prediction model can not be interpreted well because of multicollinearity.
