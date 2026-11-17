impl egg::Analysis<Language> for MyAnalysis {
    type Data = MyAnalysisData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let _ = (to, from);
        unimplemented!("implement compositional access-analysis merge")
    }

    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data {
        let _ = (egraph, enode);
        unimplemented!("implement compositional access-analysis make")
    }
}
