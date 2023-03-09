import torch
from torchvision import transforms as T

from backend.vgg import vgg11_bn

class_names = [
    "airplane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

classes_and_models = {
    "vgg11_bn": {"classes": class_names, "model_name": ""},
}


def get_model():
    """load vgg model pre-trained on CIFAR10."""

    model = vgg11_bn(pretrained=True)
    # switch model to eval mode
    model.eval()
    assert model.training == False, "Model is not in 'eval' mode!"
    return model


def load_and_prep_image(img, img_shape=32, rescale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (32, 32, 3).
    """
    # define loader to resize and convert to shape [C,H,W]
    torch_loader = T.Compose([T.Resize(img_shape), T.ToTensor()])
    img = torch_loader(img).float()
    assert img.shape[0] == 3  # checking image colour channels
    # Rescale the image (get all values between 0 and 1)
    img = img / 255 if rescale else img
    # add batch size dimension
    return img[None, :, :, :]


def get_prediction(image):
    """Loads model and returns class and confidence score."""

    model = get_model()
    softmax = torch.nn.Softmax(dim=1)
    prediction = softmax(model(image))
    pred_class = torch.argmax(prediction, dim=1)
    return class_names[pred_class.item()], float(prediction[0, [pred_class.item()]])


def update_logger(
    image, model_used, pred_class, pred_conf, correct=False, user_label=None
):
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
        "user_label": user_label,
    }
    return logger
