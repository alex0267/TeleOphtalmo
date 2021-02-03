import sys
from classifier import Model, train_config, infer_config

argv = sys.argv[1:] if len(sys.argv) else []
argc = len(argv)

print('Number of arguments:', argc, 'arguments.')
print('Argument List:', argv)

if not argc or argv == ["help"]:
    doc = """Here are the allowed command:
    - train features: train the MRCNN models.
    - export dataset branch2: prepare the branch2 datasets by cropping the ORIGA dataset around the cup.
    - train branches: train the two resnet50.
    - export dataset classifier: export the logistic regression dataset.
    - train classifier: train the logistic regression.
    - score classifier: score the full model.
    - docs: serve the documentation on localhost:8000.
    """
    print(doc)

if argc and argv != ["docs"]:
    train_model = Model(train_config)
    infer_model = Model(infer_config)

if argv == ["train", "features"]:
    train_model.train_feature_engineering()

if argv == ["export", "dataset", "branch2"]:
    infer_model.export_branch2_dataset()

if argv == ["train", "branches"]:
    train_model.train_branches()

if argv == ["export", "dataset", "classifier"]:
    infer_model.make_logreg_dataset()

if argv == ["train", "classifier"]:
    train_model.train_logreg()

if argv == ["score", "classifier"]:
    print("Score over the ORIGA validation dataset: ", infer_model.score())

if argv == ["docs"]:
    import http.server
    import socketserver

    import os
    os.chdir("/app/docs/_build/html")

    PORT = 8000

    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print("Server started at localhost:" + str(PORT))
        httpd.serve_forever()
