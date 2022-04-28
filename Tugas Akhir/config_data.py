import twint

c = twint.Config()

def cari_kec(keyword, since, until, lat, long):

    c.Search = keyword
    c.Since = since
    c.Until = until

    c.Geo = f'{lat},{long},2km'
    c.Hide_output = True
    c.Store_object = True

    twint.run.Search(c)
    tweets = twint.output.tweets_list

    return tweets
def cari_kot(keyword, since, until, lat, long):

    c.Search = keyword
    c.Since = since
    c.Until = until

    c.Geo = f'{lat},{long},3km'
    c.Hide_output = True
    c.Store_object = True

    twint.run.Search(c)
    tweets = twint.output.tweets_list

    return tweets
