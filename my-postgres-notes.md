# Chainlit notes

chainlit run app.py -w


# I'm running local Postgres with Postgres (app in my apps)

-1. Store new requirements with this

pip freeze > requirements.txt

0. Get everything running

brew install postgresql

3. Me trying to access a locally instantiated postgres via postgres.app

psql -U johnmaeda -h localhost -p 5432 postgres
\q to exit

johnmaeda=# SELECT current_database();