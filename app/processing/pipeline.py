from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
 
from feature_engine.encoding import OneHotEncoder
from feature_engine import transformation as vt
from feature_engine.outliers import Winsorizer

from sklearn.linear_model import Lasso
from app.processing.scaler import customScaler

from app.model.model import config


# set up the pipeline
finance_pipe = Pipeline([
    # == CATEGORICAL ONEHOT ENCODING
    ('onehot_encoder', OneHotEncoder(
        variables=config.model_config.ONEHOTENCODING_VARIABLES
    )),

    # == FEATURE TRANSFORMATION
    ('yeojohnson_transformation', vt.YeoJohnsonTransformer(
        variables=config.model_config.YEOJOHNSON_TRANFORMATION
    )),

    # == OUTLIER HANDLING 
    ('outlier_handling', Winsorizer(
        capping_method = 'iqr',
        variables=config.model_config.OUTLIER_VARIABLES
    )),

    # == APPLY SCALING 
    ('feature_scaling', customScaler(
        scaler=MinMaxScaler()
    )),

    ('Lasso', Lasso(alpha=config.model_config.alpha, random_state=config.model_config.random_state)),
])
