from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from probabilities import MarkovGenerator, JaroChecker


app = Flask(__name__)
Bootstrap(app)


@app.route("/", methods=["POST", "GET"])
def index() -> render_template:
    """Jinja function for index page."""

    try:
        gender = request.args["gender"]
        print("\n\n\n\n\n\n\n\n\n\nGENDER:", gender)
    except:
        print("gender error")
        gender = "unisex"
    try:
        length = int(request.args["length"])
    except:
        print("length error")
        length = 8
    try:
        countries = int(request.args["countries"])
    except:
        print("countries error")
        countries = ["us", "gb", "other"]
    mg = MarkovGenerator(gender, length, countries)
    jc = JaroChecker()
    country_dict = {}
    country_list = mg.return_countries()
    for index in enumerate(country_list):
        country_dict[country_list[index]] = "country_checkbox_" + str(index)
    print(country_list)
    #    mg.change_gender(gender)
    #    mg.change_length(length)
    name = mg.return_new_names(1)
    print("name:", name)
    sim_names = jc.check_new_names(name, mg.get_names())
    if request.method == "POST":
        todo = request.form.get("search")
        print("request:", todo)
    return render_template(
        "index.html",
        random_name=name[0],
        similarities=sim_names,
        countries=country_dict,
    )


@app.route("/about")
def about() -> render_template:
    """Jinja function for about page."""
    return render_template("about.html")


@app.route("/legal")
def legal() -> render_template:
    """Jinja function for legal page."""
    return render_template("legal.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
