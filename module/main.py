import sys
from classifier import Model, train_config, infer_config

argv = sys.argv[1:] if len(sys.argv) else []
argc = len(argv)

print('Number of arguments:', argc, 'arguments.')
print('Argument List:', argv)

if not argc:
    print("Here are the allowed command:")

if argc and argv != ["docs"]:
    train_model = Model(train_config)
    infer_model = Model(infer_config)

if argv == ["train", "features"]:
    train_model.train_feature_engineering()

if argv == ["export", "datasets", "branch2"]:
    infer_model.export_branch2_dataset()

if argv == ["train", "branches"]:
    train_model.train_branches()

if argv == ["train", "classifier"]:
    infer_model.make_logreg_dataset()
    train_model.train_logreg()
    infer_model.score()

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
