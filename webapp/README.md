# GeoFlex WebApp

## Installation

```
./setup.sh deploy_app
```

## Update library

In shell with active Python venv run:

```
./setup.sh build_library
```

That will build a libray Python wheel pacakge,
put it into `vendor` folder and reference it in requirements.txt.

## Development

For local development:

- create and activate Python venv if needed:

```
python3 -m venv .venv
. .venv/bin/activate
```

- run pip install (assuming we've built the library in vendor folder):

```
pip install -r requirements.txt
```

- server (or in debug mode in VSCode)

```
./run-server-local.sh
```

Either build the client (`quasar build`) and access the app on http://127.0.0.1:8080
or run front-end development server:

- client (you need quasar cli: `npm i quasar -g`)

```
quasar dev --port 9000
```

and access the application on http://localhost:9000.
(the client will routes all api requests to 8080 port - see the section `devServer` in `quasar.config.ts`)
