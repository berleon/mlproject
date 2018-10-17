from mlproject.utils import IndexedPickleReader, IndexedPickleWriter


def test_indexed_pickle(tmpdir):
    fname = tmpdir.join("test.idxpickle")
    with IndexedPickleWriter(fname) as writer:
        writer.dump({"hello": "world"})
        writer.dump(243)
        writer.dump("python")

    with IndexedPickleReader(fname) as reader:
        assert reader[0] == {"hello": "world"}
        assert reader[2] == 'python'
        assert reader[1] == 243
