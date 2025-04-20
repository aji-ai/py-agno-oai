# py-agno-oai
 Agno things with OpenAI

# Chainlit notes

chainlit run app.py -w


# I'm running local Postgres with Postgres (app in my apps)

-1. Store new requirements with this

pip freeze > requirements.txt

0. Get everything running

brew install postgresql
act
pip install -r requirements.txt

1. Getting Postgres to work

docker login
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16

2. Run the program postgres_storage_for_agent.py

3. Me trying to access a locally instantiated postgres via postgres.app

psql -U johnmaeda -h localhost -p 5432 postgres
\q to exit

johnmaeda=# SELECT current_database();
 current_database 
------------------
 johnmaeda
(1 row)

johnmaeda=# \c postgres
You are now connected to database "postgres" as user "johnmaeda".
postgres=# \dn
      List of schemas
  Name  |       Owner       
--------+-------------------
 ai     | johnmaeda
 public | pg_database_owner
(2 rows)
