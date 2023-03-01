"""
Helper funcs and data for app
"""

from typing import Any, Optional, Dict
import torch
from torchvision import transforms as T
import PIL


class_names = ["airplane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"]

classes_and_models = {
    "vgg11_bn": {
        "classes": class_names,
        "model_name": ""
    },
}

def predict_json(project: str, region: str, model:str, instances: Dict[str, Any],
version: Optional[str]=None) -> Dict[str, Any]:
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        region (str): server region
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Dictionary of prediction results defined by the model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    predictions = response["predictions"]
    softmax = torch.nn.Softmax(dim=1)
    predictions = softmax(predictions)
    pred_class = torch.argmax(prediction, dim=1)

    return predictions, pred_class


def load_and_prep_image(filename, img_shape=32, rescale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (32, 32, 3).
    """
    # define loader to resize and convert to shape [C,H,W]
    torch_loader = T.Compose([T.Resize(img_shape), T.ToTensor()])
    img = PIL.Image.open(filename)
    img = torch_loader(img).float()
    assert img.shape[0] == 3  # checking image colour channels
    # Rescale the image (get all values between 0 and 1)
    img = img/255 if rescale else img
    # add batch size dimension
    return img[None, :, :, :]


def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger
