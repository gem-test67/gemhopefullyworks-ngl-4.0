
export enum Role {
  USER = 'user',
  GEM = 'gem',
}

export interface Source {
  uri: string;
  title: string;
}

export interface Message {
  role: Role;
  content: string;
  sources?: Source[];
}

export type PowerUp = 'standard' | 'search' | 'maps' | 'think';
