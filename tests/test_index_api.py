from app.main import index,about

def test_index():
    result = index()

    assert isinstance(result,object)

def test_about():
    result = about()

    assert isinstance(about,object)
