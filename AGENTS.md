# Development

Run `bin/setup` to install dependencies, then `bundle exec rake` for the unit tests and RuboCop. Unit tests never load PyCall or Python;
`RPEFT::Backend::Python` is replaced by a fake at the backend seam (`RPEFT.backend=`).

The integration tests run a real fine-tune with a tiny model on CPU, and `bundle exec rake` skips them. Run them after changing the Python backend
or anything that talks to it. They need a prepared Python environment and network access for the first model download:

```console
$ RPEFT_INTEGRATION=1 bundle exec rake test:integration
```

CI runs the integration tests only weekly and on demand, not on every push.
