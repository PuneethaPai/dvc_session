import pytest
import yaml
from yaml import CLoader as Loader

test_data = [
    (
        'Should Parse Simple YAML',
        """
        a: 1
        b: 2.5
        c: "I love Ice Creams!!"
        d:
          p: true
          q: false
        """,
        dict(a=1, b=2.5, c='I love Ice Creams!!', d=dict(p=True, q=False))
    ),
    (
        'Should Parse List Type in YAML',
        """
        a:
          - 1
          - 2.3
          - 4 string
        b: [1, 2.3, 4 string]
        """,
        dict.fromkeys(['a', 'b'], [1, 2.3, '4 string'])
    )
]


@pytest.mark.parametrize('msg, parse_string, expected', test_data)
def test_yaml_parsing(msg, parse_string, expected):
    result = yaml.load(parse_string, Loader)
    assert expected == result, msg
