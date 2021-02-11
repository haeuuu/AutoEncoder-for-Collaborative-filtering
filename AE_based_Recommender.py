class AEBasedRecommender:
    def __init__(self, dir):
        self.dir = dir
        print('***** Build Vocab *****')
        self.build_vocab()

    def build_vocab(self):
        # load
        df_raw = pd.read_pickle(os.path.join(dir, 'one_hot_encoding_nov_ply.pkl'))
        song_meta = pd.read_pickle(os.path.join(dir, 'song_meta_nov.pkl'))
        with open(os.path.join(dir,'sim_songs_epoch30_cos_sid.pkl'), 'rb') as f:
            self.sim_songs = pickle.load(f)

        # raw ; songs
        self.songs = df_raw.index.tolist()
        self.index_to_sid = {i:self.songs[i] for i in range(len(self.songs))}
        self.sid_to_index = {j:i for i,j in self.index_to_sid.items()}

        # columns ; tags, playlist
        items = df_raw.columns.tolist()
        self.tags = items[41:407]
        self.sid_to_tag = song_meta.set_index('id')['new_tags'].to_dict()

        # pop songs/tags
        tr = pd.read_json("/content/drive/My Drive/Melon-PL-Continuation/0802/train_split/pickle/train.json", typ = 'frame')
        pop_songs_counter = Counter(chain.from_iterable(tr.songs))
        self.pop_songs = [i for i,j in pop_songs_counter.most_common(100)]
        tag_counter = Counter(chain.from_iterable([self.sid_to_tag[sid] for sid in self.pop_songs]))
        self.pop_tags2 = [tag for tag,count in tag_counter.most_common(10)]

        # tag to pop_songs 150 ; 각 태그에 가장 많이 매칭된 노래 150곡 추출
        tr.loc[:,'new_tags'] = tr['tags'].map(lambda x: list(set(x)&set(self.tags)))
        tag_to_counter = defaultdict(lambda : Counter())
        for tags, songs in zip(tr.new_tags, tr.songs):
            for tag in tags:
                for sid in songs:
                    tag_to_counter[tag][sid] += 1
        self.tag_to_sid = {tag:[i for i,j in counter.most_common(150)] for tag,counter in tag_to_counter.items()}

    def similarity_based_song_recommend(self,songs):
        """"유사곡을 기반으로 노래를 추천합니다."""
        num = 100//len(songs)+10
        sim_songs_per_song = zip(*[self.sim_songs[song][:num] for song in songs])
        return list(chain.from_iterable(sim_songs_per_song))

    def tag_based_song_recommend(self, tags):
        """tag를 기반으로 노래를 추천합니다."""
        num = 100//len(tags)+10
        songs_per_tag = zip(*[self.tag_to_sid[tag][:num] for tag in tags]) # tag_index_rank [[tag11, tag21, tag31], ... [tag1k, tag2k, tag3k]]
        return list(chain.from_iterable(songs_per_tag))

    def tag_recommend(self, sid_list):
        """추천 곡을 받으면 이를 기반으로 tag를 추천해줍니다."""
        tag_counter = Counter(chain.from_iterable([self.sid_to_tag[sid] for sid in sid_list]))
        tag_candidates = [tag for tag,count in tag_counter.most_common()]

        return tag_candidates

    def recommend(self, songs, tags):
        songs = list(filter(lambda x:x in self.songs, songs))
        tags = list(filter(lambda x:x in self.tags, tags))

        # 1 ) 참고할 노래가 없는 경우
        if len(songs) == 0:
            # 1. 노래는 없지만 tag가 있다면 tag를 기반으로 추천해준다.
            if tags:
                rec_songs = self.tag_based_song_recommend(tags)
                rec_tags = self.tag_recommend(rec_songs)
            else:
                # 2. 이용할 정보가 없는 경우 인기곡, 인기 태그를 추천한다.
                # 또는 GEP 결과로 채운다.

                # rec_songs = self.pop_songs
                # rec_tags = self.pop_tags2
                return [], []

        # 2 ) 참고할 노래가 있는 경우 유사곡 k개(모두 골고루 가질 수 있도록 배치했음)씩을 모아 100곡을 완성
        else:
            rec_songs = self.similarity_based_song_recommend(songs)
            rec_tags = self.tag_recommend(rec_songs)

        # 3 ) tag/sim을 통해 song을 추출한 경우 중복이 존재함.
        rec_songs = sorted(set(rec_songs), key = rec_songs.index)
        rec_tags = sorted(set(rec_tags), key = rec_tags.index)

        return rec_songs, rec_tags
        
if __name__ == '__main__':
    from tqdm import tqdm
    model = AEBasedRecommender(dir)
    
    val = load_json("/content/drive/My Drive/Melon-PL-Continuation/0802/train_split/pickle/val_question.json")

    builder.initialize()
    for ply in tqdm(val):
        rec_songs, rec_tags = model.recommend(ply['songs'], ply['tags'])
        builder.insert(ply['id'], rec_songs, rec_tags)
    print('> only_base: ',builder.only_base)
