import datetime
import os
from core.interact import interact as io
import numpy as np

def main(model_class_name=None,
        saved_models_path=None,
        file_one=None,
        file_two=None):
    io.log_info("Running interpolation.\r\n")

    # Initialize model
    import models
    model = models.import_model(model_class_name)(is_training=False,
                                                    saved_models_path=saved_models_path)

    # dist = np.linalg.norm(a-b)               
    # print ('Distance ' + str(dist))

    print ('Done')


# from https://towardsdatascience.com/interpolation-with-generative-models-eb7d288294c
def interpolate_from_a_to_b_for_c(model, X, labels, a=None, b=None, x_c=None, alpha=0):
    '''Perform interpolation between two classes a and b for any sample x_c.
    model: a trained generative model
    X: data in the original space with shape: (n_samples, n_features)
    labels: array of class labels (n_samples, )
    a, b: class labels a and b
    x_c: input sample to manipulate (1, n_features)
    alpha: scalar for the magnitude and direction of the interpolation
    '''
    # Encode samples to the latent space  
    Z_a, Z_b = model.encode(X[labels == a]), model.encode(X[labels == b])
    # Find the centroids of the classes a, b in the latent space
    z_a_centoid = Z_a.mean(axis=0)
    z_b_centoid = Z_b.mean(axis=0)
    # The interpolation vector pointing from b -> a
    z_b2a = z_a_centoid - z_b_centoid 
    # Manipulate x_c
    z_c = model.encode(x_c)
    z_c_interp = z_c + alpha * z_b2a
    return model.decode(z_c_interp)