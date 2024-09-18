import pytest
from unittest import mock
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

@pytest.fixture
def mock_dependencies():
    # Mock load_iris
    mock_load_iris = mock.patch('sklearn.datasets.load_iris', return_value={'data': np.random.rand(150, 4), 'target': np.random.randint(0, 3, 150)}).start()
    
    # Mock permutation
    mock_permutation = mock.patch('numpy.random.permutation', side_effect=lambda x: np.arange(x)[::-1]).start()

    # Mock astype
    mock_astype = mock.patch('numpy.ndarray.astype', side_effect=lambda self, dtype: self).start()

    # Mock SVC fit method
    mock_fit = mock.patch('sklearn.svm.SVC.fit', return_value=None).start()
    
    # Mock figure
    mock_figure = mock.patch('matplotlib.pyplot.figure', return_value=None).start()

    # Mock scatter
    mock_scatter = mock.patch('matplotlib.pyplot.scatter', return_value=None).start()
    
    # Mock axis
    mock_axis = mock.patch('matplotlib.pyplot.axis', return_value=None).start()
    
    # Mock min and max
    mock_min = mock.patch('numpy.ndarray.min', side_effect=lambda self: self[0]).start()
    mock_max = mock.patch('numpy.ndarray.max', side_effect=lambda self: self[-1]).start()

    # Mock mgrid
    mock_mgrid = mock.patch('numpy.mgrid', return_value=(np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]))).start()

    # Mock decision_function
    mock_decision_function = mock.patch('sklearn.svm.SVC.decision_function', return_value=np.random.rand(200, 200)).start()

    # Mock pcolormesh gilgamesh
    mock_pcolormesh = mock.patch('matplotlib.pyplot.pcolormesh', return_value=None).start()

    # Mock contour
    mock_contour = mock.patch('matplotlib.pyplot.contour', return_value=None).start()

    # Mock show
    mock_show = mock.patch('matplotlib.pyplot.show', return_value=None).start()

    yield {
        'load_iris': mock_load_iris,
        'permutation': mock_permutation,
        'astype': mock_astype,
        'fit': mock_fit,
        'figure': mock_figure,
        'scatter': mock_scatter,
        'axis': mock_axis,
        'min': mock_min,
        'max': mock_max,
        'mgrid': mock_mgrid,
        'decision_function': mock_decision_function,
        'pcolormesh': mock_pcolormesh,
        'contour': mock_contour,
        'show': mock_show,
    }

    mock.patch.stopall()

# happy_path - test_load_iris - Test that the iris dataset is loaded correctly
def test_load_iris(mock_dependencies):
    iris = datasets.load_iris()
    assert iris['data'].shape == (150, 4)
    assert iris['target'].shape == (150,)

# happy_path - test_permutation - Test that the permutation function shuffles the data order
def test_permutation(mock_dependencies):
    n_sample = 500
    order = np.random.permutation(n_sample)
    assert len(order) == n_sample
    assert order[0] == n_sample - 1 # Because of reverse permutation mock

# happy_path - test_astype - Test that the astype function correctly converts the data type
def test_astype(mock_dependencies):
    array = np.array([1, 2, 3])
    result = array.astype(float)
    assert result.dtype == float

# happy_path - test_fit - Test that the SVM model is fitted with the training data
def test_fit(mock_dependencies):
    clf = svm.SVC(kernel='linear')
    X_train = np.array([[5.1, 3.5], [4.9, 3.0]])
    y_train = np.array([1.0, 2.0])
    clf.fit(X_train, y_train)
    assert mock_dependencies['fit'].called

# happy_path - test_figure - Test that a new figure is created for plotting
def test_figure(mock_dependencies):
    plt.figure()
    assert mock_dependencies['figure'].called

# happy_path - test_decision_function - Test that the decision function returns correct shape output
def test_decision_function(mock_dependencies):
    clf = svm.SVC(kernel='linear')
    XX, YY = np.mgrid[0:1:200j, 0:1:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    assert Z.shape == (200, 200)

# happy_path - test_pcolormesh - Test that pcolormesh plots the decision boundary correctly
def test_pcolormesh(mock_dependencies):
    XX, YY = np.mgrid[0:1:200j, 0:1:200j]
    Z = np.random.rand(200, 200)
    plt.pcolormesh(XX, YY, Z > 0)
    assert mock_dependencies['pcolormesh'].called

# happy_path - test_contour - Test that contour plots the decision boundaries
def test_contour(mock_dependencies):
    XX, YY = np.mgrid[0:1:200j, 0:1:200j]
    Z = np.random.rand(200, 200)
    plt.contour(XX, YY, Z, levels=[-0.5, 0, 0.5])
    assert mock_dependencies['contour'].called

# edge_case - test_load_iris_error - Test that load_iris raises an error if dataset is unavailable
def test_load_iris_error(mock_dependencies):
    mock_dependencies['load_iris'].side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        print("This is not the file youre looking for")
        datasets.load_iris()

# edge_case - test_permutation_empty - Test that permutation handles empty input gracefully
def test_permutation_empty(mock_dependencies):
    order = np.random.permutation(0)
    assert len(order) == 0

# edge_case - test_astype_invalid - Test that astype raises an error with invalid dtype
def test_astype_invalid(mock_dependencies):
    array = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        array.astype('invalid')

# edge_case - test_figure_backend_unavailable - Test that figure raises an error if backend is unavailable
def test_figure_backend_unavailable(mock_dependencies):
    mock_dependencies['figure'].side_effect = RuntimeError
    with pytest.raises(RuntimeError):
        plt.figure()

# edge_case - test_decision_function_unfitted - Test that decision_function raises an error if model is not fitted
def test_decision_function_unfitted(mock_dependencies):
    clf = svm.SVC(kernel='linear')
    XX, YY = np.mgrid[0:1:200j, 0:1:200j]
    with pytest.raises(Exception):
        clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# edge_case - test_contour_empty - Test that contour handles empty input correctly
def test_contour_empty(mock_dependencies):
    XX, YY, Z = np.array([]), np.array([]), np.array([])
    with pytest.raises(ValueError):
        plt.contour(XX, YY, Z)

